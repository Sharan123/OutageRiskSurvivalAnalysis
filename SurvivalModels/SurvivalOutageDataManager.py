"""
SurvivalOutageDataManager
=========================

Reformulates power line outage prediction as a survival analysis problem.

Instead of:  "Will line L fail in the next 6 hours?" (binary, 1:20,000+ imbalance)
We ask:      "What is the hazard rate for line L given current conditions?"
             (continuous, no class imbalance)

Key concepts:
    Spell:    A period of healthy operation for a specific line, from the moment
              it becomes healthy until it either fails (event) or the observation
              ends (censored). Every line has multiple spells across the timeline (they might not have any outages though).

    Event:    Weather-related outage onset (transition from 0→1 in outage_matrix).
              This is the "death" in survival terminology.

    Censoring: The spell ends without a weather outage because:
              - The observation window (split boundary) is reached
              - A non-weather outage occurs (competing risk)
              - The line was already in outage at the start of observation

    Covariates: Time-varying weather + static line features. Updated hourly.

This manager wraps an existing OutageDataManager and provides:
    1. Spell extraction from the (Time × Lines) outage matrix
    2. Discrete-time survival labels compatible with:
       - Classical: Cox PH, Weibull AFT (via lifelines/pycox)
       - Deep: DeepSurv, DeepHit, Dynamic-DeepHit (via pycox)
       - GNN-based: Your existing GNN+LSTM with modified loss function
    3. A construct_sample method that returns HeteroData with survival targets

Compatible models (in order of recommended priority):
    A) GNN+LSTM with discrete hazard output (your existing architecture, new loss)
    B) Dynamic-DeepHit via pycox (handles time-varying covariates natively)
    C) Random Survival Forest (scikit-survival) as a strong tabular baseline
    D) Cox PH with time-varying covariates (lifelines)
"""

import os
import pickle
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ============================================================================ #
#  Data Structures                                                             #
# ============================================================================ #

@dataclass
class Spell:
    """A single period of healthy operation for one line."""
    line_idx: int           # Index into line_ids
    t_start: int            # First hour the line is healthy (spell begins)
    t_end: int              # Last hour of the spell (exclusive)
    event: bool             # True = weather outage onset at t_end; False = censored
    duration: int           # t_end - t_start (in hours)
    censor_reason: str      # 'event', 'split_boundary', 'non_weather_outage', 'end_of_data'

    @property
    def t_event_or_censor(self):
        return self.t_end


# ============================================================================ #
#  Survival Dataset                                                            #
# ============================================================================ #

class SurvivalDataset(Dataset):
    """
    PyTorch Dataset that yields survival-formatted HeteroData samples.

    Each sample is anchored at a timestep t and contains:
        - Input features: same as original (weather, line dynamic, line static, edges)
        - Survival targets per line:
            - time_to_event: hours until next weather outage (or censoring)
            - event_indicator: 1 if the line actually fails, 0 if censored
            - discrete_hazard: per-interval hazard labels for discrete-time models
    """
    def __init__(self, manager, mode):
        self.manager = manager
        self.mode = mode

    @property
    def indices(self):
        return self.manager.split_indices[self.mode]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_start = self.indices[idx]
        return self.manager.construct_sample(t_start)


# ============================================================================ #
#  Main Manager                                                                #
# ============================================================================ #

class SurvivalOutageDataManager:
    """
    Wraps an existing OutageDataManager and adds survival analysis targets.

    The original manager handles:
        - Weather data loading (HDF5)
        - Graph structure (weather→line edges, line↔line edges)
        - Static line features
        - Outage matrix construction

    This manager adds:
        - Spell extraction per line
        - Survival-formatted labels (time-to-event, censoring, discrete hazard)
        - Modified sampling that uses ALL timesteps (no positive/negative distinction)
        - Modified construct_sample with survival targets

    Args:
        base_manager: Initialized OutageDataManager instance
        max_horizon: Maximum prediction horizon in hours. Events beyond this
                     are right-censored at max_horizon. (default: 72 = 3 days)
        discrete_intervals: Number of discrete time bins for hazard estimation.
                           (default: 12, meaning 6-hour bins over 72-hour horizon)
        train_stride: Stride for training set sampling. Since survival analysis
                     uses ALL timesteps (not just near-outage ones), stride > 1
                     reduces dataset size without losing events. (default: 1)
        val_stride: Stride for validation sampling. (default: 1)
        test_stride: Stride for test sampling. (default: 1, dense for evaluation)
    """

    def __init__(
        self,
        base_manager,
        max_horizon: int = 1008,
        discrete_intervals: int = 6,
        train_stride: int = 1,
        val_stride: int = 1,
        test_stride: int = 1,
        verbose: bool = True,
        processed_dir: str = "SurvivalModels/data/processed",
    ):
        self.base = base_manager
        self.max_horizon = max_horizon
        self.discrete_intervals = discrete_intervals
        self.train_stride = train_stride
        self.val_stride = val_stride
        self.test_stride = test_stride
        self.verbose = verbose
        
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        # Interval boundaries for discrete-time hazard
        # e.g., max_horizon=1008, intervals=6 → bins of 168 hours/1 week each
        self.interval_width = max_horizon / discrete_intervals
        self.interval_bounds = np.linspace(0, max_horizon, discrete_intervals + 1)

        # Forward attributes from base manager
        self._forward_base_attributes()

        # Extract spells and build survival labels
        self.spells = {}           # {split_name: list of Spell}
        self.spell_index = None    # Pre-built (Time, Line) → survival target lookup
        self.split_indices = {}
        
        # Configuration Hash for caching
        config_str = f"{self.base.config_hash}_{max_horizon}_{discrete_intervals}_{train_stride}_{val_stride}_{test_stride}"
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()
        self.meta_path = os.path.join(self.processed_dir, f"survival_meta_{self.config_hash}.pkl")

        self._initialize_dataset()

        if self.verbose:
            self._print_summary()

    def _initialize_dataset(self):
        """Main orchestration method to load or build the dataset metadata."""
        rebuild = False
        if os.path.exists(self.meta_path):
            if self.verbose: print(f"Loading cached survival metadata from {self.meta_path}...")
            with open(self.meta_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'tte_matrix' not in data or 'event_matrix' not in data:
                print("Cached dataset missing required metadata. Rebuilding...")
                rebuild = True
            else:
                for k, v in data.items():
                    setattr(self, k, v)
        else:
            rebuild = True

        if rebuild:
            if self.verbose: print("Building survival dataset from source...")
            self._extract_all_spells()
            self._build_time_to_event_matrix()
            self._generate_survival_indices()
            
            # Cache the metadata
            save_dict = {
                'spells': self.spells,
                'split_indices': self.split_indices,
                'tte_matrix': self.tte_matrix,
                'event_matrix': self.event_matrix
            }
            with open(self.meta_path, 'wb') as f:
                pickle.dump(save_dict, f)

    def _forward_base_attributes(self):
        """Forward commonly accessed attributes from the base manager."""
        # These are used by construct_sample and external code
        self.line_ids = self.base.line_ids
        self.line_features_static = self.base.line_features_static
        self.line_xref_idx = self.base.line_xref_idx
        self.outage_matrix = self.base.outage_matrix
        self.all_outages_matrix = self.base.all_outages_matrix
        self.time_since_matrix = self.base.time_since_matrix
        self.weather_edges_index = self.base.weather_edges_index
        self.weather_edges_attr = self.base.weather_edges_attr
        self.line_edges_index = self.base.line_edges_index
        self.line_edges_attr = self.base.line_edges_attr
        self.input_win = self.base.input_win
        self.output_win = self.base.output_win
        self.train_end_idx = self.base.train_end_idx
        self.val_end_idx = self.base.val_end_idx
        self.unique_weather_cells = self.base.unique_weather_cells
        self.weather_preprocessor = self.base.weather_preprocessor

    # ------------------------------------------------------------------ #
    #  Spell Extraction                                                   #
    # ------------------------------------------------------------------ #

    def _extract_all_spells(self):
        """
        Extract survival spells from the outage matrix for each line.

        A spell is a contiguous period where the line is NOT in a weather outage.

        Timeline for one line:
            0000000001111100000000000001110000000...
                     ^    ^            ^  ^
                     |    |            |  |
                  onset  recovery   onset recovery

        Spells:  [---A---][-----B-------][--C--]...
            A: t_start=0, t_end=9, event=True (weather outage at t=9)
            B: t_start=14, t_end=27, event=True (weather outage at t=27)
            C: t_start=30, t_end=..., depends on what comes next

        Censoring rules:
            - If spell reaches a split boundary → censored at boundary
            - If a non-weather outage starts during spell → censored
            - If data ends → censored
            - If weather outage onset → EVENT (not censored)
            TODO: Question - What if no outage at all for a line? One long censored spell from start to end? How do we learn that no outage will occur?
        """
        om = self.outage_matrix         # Weather outages (labels)
        aom = self.all_outages_matrix   # All outages (for competing risk censoring)
        total_t, n_lines = om.shape

        # Precompute weather outage onsets: 0→1 transitions
        weather_onset = np.diff(om, axis=0, prepend=0)  # (T, N_lines)
        # onset[t, l] == 1 means weather outage STARTS at hour t for line l

        # Split boundaries
        boundaries = {
            'train': (0, self.train_end_idx),
            'val': (self.train_end_idx, self.val_end_idx),
            'test': (self.val_end_idx, total_t),
        }

        all_spells = defaultdict(list)

        for split_name, (t_split_start, t_split_end) in boundaries.items():
            if self.verbose:
                print(f"Extracting spells for {split_name} "
                      f"[{t_split_start}, {t_split_end})...")

            for line_idx in range(n_lines):
                # Extract weather outage column for this line in this split
                line_weather = om[t_split_start:t_split_end, line_idx]
                line_all = aom[t_split_start:t_split_end, line_idx]
                line_onset = weather_onset[t_split_start:t_split_end, line_idx]

                split_len = t_split_end - t_split_start
                t = 0

                while t < split_len:
                    # Skip if currently in ANY outage (not a spell start)
                    if line_weather[t] == 1 or line_all[t] == 1:
                        t += 1
                        continue

                    # Start of a healthy spell
                    spell_start_local = t
                    spell_start_global = t_split_start + t

                    # Walk forward to find spell end
                    t += 1
                    event = False
                    censor_reason = 'end_of_split'

                    while t < split_len:
                        # Check: weather outage onset?
                        if line_onset[t] == 1:
                            event = True
                            censor_reason = 'event'
                            break

                        # Check: non-weather outage starts?
                        # (line goes to outage in all_outages but NOT in weather)
                        if line_all[t] == 1 and line_weather[t] == 0:
                            censor_reason = 'non_weather_outage'
                            break

                        t += 1

                    spell_end_global = t_split_start + t
                    duration = t - spell_start_local

                    # Only keep spells with duration > 0
                    if duration > 0:
                        spell = Spell(
                            line_idx=line_idx,
                            t_start=spell_start_global,
                            t_end=spell_end_global,
                            event=event,
                            duration=duration,
                            censor_reason=censor_reason,
                        )
                        all_spells[split_name].append(spell)

                    # Advance past the outage period
                    if t < split_len and (line_weather[t] == 1 or line_all[t] == 1):
                        while t < split_len and (line_weather[t] == 1 or line_all[t] == 1):
                            t += 1

        self.spells = dict(all_spells)

        if self.verbose:
            for split, sp_list in self.spells.items():
                n_events = sum(1 for s in sp_list if s.event)
                n_censored = sum(1 for s in sp_list if not s.event)
                
                # Durations of all spells
                durations = [s.duration for s in sp_list]
                
                # Durations excluding non-weather outages
                valid_durations = [s.duration for s in sp_list if s.censor_reason != 'non_weather_outage']
                
                print(f"  {split}: {len(sp_list)} spells, "
                      f"{n_events} events, {n_censored} censored")
                if durations:
                    print(f"    All spells: median duration={np.median(durations):.0f}h, mean={np.mean(durations):.0f}h")
                if valid_durations:
                    print(f"    Excl. non-weather outages: median duration={np.median(valid_durations):.0f}h, mean={np.mean(valid_durations):.0f}h")

    # ------------------------------------------------------------------ #
    #  Time-to-Event Matrix                                               #
    # ------------------------------------------------------------------ #

    def _build_time_to_event_matrix(self):
        """
        Pre-compute per-(timestep, line) survival targets.

        For each (t, line_idx) where the line is healthy at time t:
            - time_to_event: hours until the next weather outage onset
                             (capped at max_horizon, right-censored if no event)
            - event: 1 if a weather outage occurs within max_horizon, 0 otherwise

        Stored as two matrices: (Total_T, N_lines) each.
        This avoids recomputing spell lookups per sample.
        """
        if self.verbose:
            print("Building time-to-event matrices...")

        om = self.outage_matrix
        aom = self.all_outages_matrix
        total_t, n_lines = om.shape

        # Initialize: time_to_event = max_horizon (all censored by default)
        self.tte_matrix = np.full((total_t, n_lines), self.max_horizon, dtype=np.float32)
        self.event_matrix = np.zeros((total_t, n_lines), dtype=np.int8)

        weather_onset = np.diff(om, axis=0, prepend=0) == 1
        all_onset = np.diff(aom, axis=0, prepend=0) == 1
        non_weather_onset = all_onset & ~weather_onset

        for line_idx in tqdm(range(n_lines), desc="Building TTE",
                             disable=not self.verbose):
            w_onsets = np.where(weather_onset[:, line_idx])[0]
            nw_onsets = np.where(non_weather_onset[:, line_idx])[0]

            w_ptr = 0
            nw_ptr = 0

            for t in range(total_t):
                # Determine split boundary for current t
                if t < self.train_end_idx:
                    split_end = self.train_end_idx
                elif t < self.val_end_idx:
                    split_end = self.val_end_idx
                else:
                    split_end = total_t

                # If the line is currently in ANY outage, it is not at risk.
                # We set tte to 0 and event to 0 (censored immediately).
                if aom[t, line_idx] == 1:
                    self.tte_matrix[t, line_idx] = 0.0
                    self.event_matrix[t, line_idx] = 0
                    continue

                # Advance pointers past current time
                while w_ptr < len(w_onsets) and w_onsets[w_ptr] < t:
                    w_ptr += 1
                while nw_ptr < len(nw_onsets) and nw_onsets[nw_ptr] < t:
                    nw_ptr += 1

                next_w = w_onsets[w_ptr] if w_ptr < len(w_onsets) else float('inf')
                next_nw = nw_onsets[nw_ptr] if nw_ptr < len(nw_onsets) else float('inf')

                # The observation ends at the earliest of: next weather outage, next non-weather outage, or split boundary
                end_time = min(next_w, next_nw, split_end)

                tte = end_time - t

                if end_time == next_w and end_time < split_end:
                    # Event observed
                    if tte < self.max_horizon:
                        self.tte_matrix[t, line_idx] = float(tte)
                        self.event_matrix[t, line_idx] = 1
                    else:
                        self.tte_matrix[t, line_idx] = float(self.max_horizon)
                        self.event_matrix[t, line_idx] = 0
                else:
                    # Censored (by non-weather outage or split boundary)
                    if tte < self.max_horizon:
                        self.tte_matrix[t, line_idx] = float(tte)
                    else:
                        self.tte_matrix[t, line_idx] = float(self.max_horizon)
                    self.event_matrix[t, line_idx] = 0

        if self.verbose:
            n_events_total = int(self.event_matrix.sum())
            n_total = total_t * n_lines
            print(f"  TTE matrix: {n_events_total} event observations "
                  f"out of {n_total} total ({100*n_events_total/n_total:.4f}%)")

    # ------------------------------------------------------------------ #
    #  Discrete Hazard Labels                                             #
    # ------------------------------------------------------------------ #

    def _compute_discrete_hazard(self, tte, event):
        """
        Convert continuous time-to-event into discrete-time hazard labels.

        For a model predicting P(event in interval k | survived to interval k),
        we need labels that indicate WHICH interval the event falls in.

        Args:
            tte: float, time-to-event in hours
            event: int, 1 if event occurred, 0 if censored

        Returns:
            labels: np.ndarray of shape (discrete_intervals,)
                    For uncensored: 1 in the interval containing the event, 0 elsewhere
                    For censored: 0 everywhere (contributes to survival likelihood)
            mask: np.ndarray of shape (discrete_intervals,)
                  1 for intervals where the subject is "at risk" (alive), 0 after
                  event or censoring. This is critical for the loss function.
        """
        labels = np.zeros(self.discrete_intervals, dtype=np.float32)
        mask = np.zeros(self.discrete_intervals, dtype=np.float32)

        for k in range(self.discrete_intervals):
            interval_start = self.interval_bounds[k]
            interval_end = self.interval_bounds[k + 1]

            if event:
                # Subject is at risk if they haven't had the event before this interval starts
                is_at_risk = tte >= interval_start
                if is_at_risk:
                    # Subject is still at risk in this interval
                    mask[k] = 1.0
                    # Event occurs in this interval: [interval_start, interval_end)
                    # We use < interval_end so intervals are strictly non-overlapping
                    if tte < interval_end:
                        labels[k] = 1.0
            else:
                # Censored
                if tte >= interval_end:
                    # Subject survived the entire interval
                    mask[k] = 1.0
                # If tte < interval_end, we don't know if it would have survived the full interval,
                # so we leave mask[k] = 0.0 to avoid optimistic bias.

        return labels, mask

    # ------------------------------------------------------------------ #
    #  Sampling Indices                                                   #
    # ------------------------------------------------------------------ #

    def _generate_survival_indices(self):
        """
        Generate timestep indices for each split.

        Unlike classification where we need positive/negative sampling,
        survival analysis uses ALL timesteps (or a strided subset).
        Every timestep contributes to the likelihood:
            - Events contribute to both the hazard and survival components
            - Censored observations contribute only to the survival component

        No class imbalance handling needed.
        """
        total_t = self.outage_matrix.shape[0]
        window_len = self.input_win + self.max_horizon  # Need input window + max horizon to avoid peeking across splits

        # Train: [0, train_end - input_win - max_horizon)
        train_stop = self.train_end_idx - window_len
        train_stop = max(0, min(train_stop, total_t - window_len))

        # Val: [train_end, val_end - input_win - max_horizon)
        val_start = self.train_end_idx
        val_stop = self.val_end_idx - window_len
        val_stop = max(val_start, min(val_stop, total_t - window_len))

        # Test: [val_end, total_t - input_win - max_horizon)
        test_start = self.val_end_idx
        test_stop = total_t - window_len

        self.split_indices = {
            'train': list(range(0, train_stop, self.train_stride)),
            'val': list(range(val_start, val_stop, self.val_stride)),
            'test': list(range(test_start, test_stop, self.test_stride)),
        }

        if self.verbose:
            for split, indices in self.split_indices.items():
                print(f"  {split}: {len(indices)} samples (stride={getattr(self, f'{split}_stride', '?')})")

    # ------------------------------------------------------------------ #
    #  Sample Construction                                                #
    # ------------------------------------------------------------------ #

    def construct_sample(self, t_start):
        """
        Construct a HeteroData sample with survival analysis targets.

        Input features are IDENTICAL to the original OutageDataManager.
        Only the targets change.

        Original targets:
            data['line'].y = binary (N_lines, 1) — outage in next 6h?
            data['line'].is_active = mask (N_lines, 1)

        Survival targets:
            data['line'].tte = (N_lines, 1) — hours until next weather outage
            data['line'].event = (N_lines, 1) — 1 if event within horizon, 0 if censored
            data['line'].discrete_labels = (N_lines, K) — per-interval event indicator
            data['line'].discrete_mask = (N_lines, K) — per-interval at-risk indicator
        """
        t_predict = t_start + self.input_win

        data = HeteroData()

        # ── 1. Node counts ──
        n_lines = len(self.line_ids)
        data['line'].num_nodes = n_lines

        # ── 2. Static line features (unchanged) ──
        data['line'].x_static = self.line_features_static  # (N_lines, F_static)
        data['line'].xref_idx = self.line_xref_idx          # (N_lines,)

        # ── 3. Dynamic line features (unchanged) ──
        status_hist = self.outage_matrix[t_start:t_predict, :]       # (T_in, N_lines)
        time_since_hist = self.time_since_matrix[t_start:t_predict, :]

        combined_dynamic = np.stack([status_hist, time_since_hist], axis=-1)
        data['line'].x_dynamic = torch.from_numpy(combined_dynamic).float().permute(1, 0, 2)

        # ── 4. Weather features (unchanged) ──
        weather_feats = self.base.get_weather_tensor_for_window(t_start, self.input_win)
        data['weather'].x = weather_feats.permute(1, 0, 2)
        data['weather'].num_nodes = data['weather'].x.size(0)

        # ── 5. Edges (unchanged) ──
        data['weather', 'affects', 'line'].edge_index = self.weather_edges_index
        data['weather', 'affects', 'line'].edge_attr = self.weather_edges_attr
        data['line', 'connected_to', 'line'].edge_index = self.line_edges_index
        data['line', 'connected_to', 'line'].edge_attr = self.line_edges_attr

        # ── 6. SURVIVAL TARGETS (the new part) ──

        # Time-to-event and event indicator at t_predict
        tte = self.tte_matrix[t_predict, :].copy()        # (N_lines,)
        event = self.event_matrix[t_predict, :].copy()     # (N_lines,)

        # Discrete hazard labels
        discrete_labels = np.zeros((n_lines, self.discrete_intervals), dtype=np.float32)
        discrete_mask = np.zeros((n_lines, self.discrete_intervals), dtype=np.float32)

        for i in range(n_lines):
            discrete_labels[i], discrete_mask[i] = self._compute_discrete_hazard(
                tte[i], event[i]
            )

        data['line'].tte = torch.from_numpy(tte).float().unsqueeze(-1)           # (N_lines, 1)
        data['line'].event = torch.from_numpy(event).float().unsqueeze(-1)       # (N_lines, 1)
        data['line'].discrete_labels = torch.from_numpy(discrete_labels).float() # (N_lines, K)
        data['line'].discrete_mask = torch.from_numpy(discrete_mask).float()     # (N_lines, K)

        # ── 7. Also provide the original binary label for comparison ──
        t_end = t_predict + self.output_win
        if t_end <= self.outage_matrix.shape[0]:
            future_status = self.outage_matrix[t_predict:t_end, :]
            y_binary = np.max(future_status, axis=0)
        else:
            y_binary = np.zeros(n_lines, dtype=np.float32)
        data['line'].y_binary = torch.from_numpy(y_binary).float().unsqueeze(-1)

        return data

    # ------------------------------------------------------------------ #
    #  DataLoader Creation                                                #
    # ------------------------------------------------------------------ #

    def get_dataset(self, mode='train'):
        if mode not in self.split_indices:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.split_indices.keys())}")
        return SurvivalDataset(self, mode)

    def create_dataloaders(self, batch_size=32, num_workers=4,
                           pin_memory=True, prefetch_factor=2):
        """
        Create DataLoaders for survival analysis.

        No stratified sampling needed — every sample contributes equally
        to the survival likelihood.
        """
        loader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0 and prefetch_factor is not None:
            loader_args["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(
            self.get_dataset('train'), shuffle=True, **loader_args
        )
        val_loader = DataLoader(
            self.get_dataset('val'), shuffle=False, **loader_args
        )
        test_loader = DataLoader(
            self.get_dataset('test'), shuffle=False, **loader_args
        )

        return train_loader, val_loader, test_loader

    def compute_interval_weights(self, train_loader, cap=50.0):
        """
        Computes the alpha weights for the DiscreteSurvivalLoss based on the
        empirical hazard rate in the training set.
        
        Args:
            train_loader: DataLoader for the training set.
            cap: Maximum allowed weight to prevent instability from extremely rare events.
            
        Returns:
            alpha: torch.Tensor of shape (K,) containing the weights per interval.
        """
        weights_path = os.path.join(self.processed_dir, f"interval_weights_{self.config_hash}.pt")
        if os.path.exists(weights_path):
            if self.verbose: print(f"Loading cached interval weights from {weights_path}...")
            return torch.load(weights_path)
            
        at_risk = torch.zeros(self.discrete_intervals)
        events = torch.zeros(self.discrete_intervals)
        
        for batch in tqdm(train_loader, desc="Computing interval weights"):
            mask = batch['line'].discrete_mask     # (B * N_lines, K)
            labels = batch['line'].discrete_labels # (B * N_lines, K)
            
            valid = mask  # Only count active lines
            at_risk += valid.sum(dim=0)
            events += (labels * valid).sum(dim=0)
        
        # alpha_k = sqrt(at_risk_k / events_k), capped
        # We use sqrt to dampen the extreme weights (e.g., sqrt(25000) ≈ 158 instead of 25000)
        alpha = torch.sqrt(at_risk / events.clamp(min=1))
        alpha = alpha.clamp(max=cap)
        
        if self.verbose:
            print("\nInterval weights (alpha):")
            for k in range(self.discrete_intervals):
                start_h = self.interval_bounds[k]
                end_h = self.interval_bounds[k+1]
                print(f"  [{start_h:.0f}-{end_h:.0f}h): at_risk={at_risk[k]:.0f}, "
                      f"events={events[k]:.0f}, alpha={alpha[k]:.2f}")
                      
        torch.save(alpha, weights_path)
        
        return alpha  # Shape: (K,)

    # ------------------------------------------------------------------ #
    #  Tabular Export (for lifelines, scikit-survival, pycox)             #
    # ------------------------------------------------------------------ #

    def export_tabular(self, split='train', include_weather_summary=False):
        """
        Export spell data as a flat DataFrame for classical survival models.

        Each row = one spell for one line.
        Columns: line_idx, duration, event, + static features, + optional
                 weather summary at spell start.

        Compatible with:
            - lifelines: CoxPHFitter, WeibullAFTFitter
            - scikit-survival: RandomSurvivalForest
            - pycox: DeepSurv, DeepHit (need further formatting)

        Returns:
            pd.DataFrame with columns:
                line_idx, duration, event, voltage, length, type_L, type_T, type_TT,
                [weather features if requested]
        """
        import pandas as pd

        if split not in self.spells:
            raise ValueError(f"No spells for split '{split}'")

        records = []
        static = self.line_features_static.numpy()  # (N_lines, 5)
        static_names = ['voltage', 'length', 'type_L', 'type_T', 'type_TT']

        for spell in self.spells[split]:
            row = {
                'line_idx': spell.line_idx,
                'duration': spell.duration,
                'event': int(spell.event),
                'censor_reason': spell.censor_reason,
                't_start': spell.t_start,
                't_end': spell.t_end,
            }

            # Static features
            for i, name in enumerate(static_names):
                row[name] = static[spell.line_idx, i]

            # Time-since-last at spell start
            if spell.t_start < self.time_since_matrix.shape[0]:
                row['time_since_last'] = float(self.time_since_matrix[spell.t_start, spell.line_idx])

            # Optional: weather summary at spell start
            # (requires weather data access — placeholder for now)
            if include_weather_summary:
                # You would call:
                # weather = self.base.get_weather_tensor_for_window(spell.t_start, 12)
                # row['max_wind'] = weather[..., wind_idx].max().item()
                # etc.
                pass

            records.append(row)

        df = pd.DataFrame(records)

        if self.verbose:
            print(f"\nTabular export for '{split}':")
            print(f"  {len(df)} spells, {df['event'].sum()} events, "
                  f"{len(df) - df['event'].sum()} censored")
            print(f"  Median duration: {df['duration'].median():.0f}h")
            print(f"  Event rate: {df['event'].mean():.4f}")

        return df

    # ------------------------------------------------------------------ #
    #  Summary & Diagnostics                                              #
    # ------------------------------------------------------------------ #

    def _print_summary(self):
        """Print a summary of the survival-formatted dataset."""
        print("\n" + "=" * 65)
        print("  SURVIVAL OUTAGE DATA MANAGER — Summary")
        print("=" * 65)
        print(f"  Lines: {len(self.line_ids)}")
        print(f"  Max horizon: {self.max_horizon}h")
        print(f"  Discrete intervals: {self.discrete_intervals} "
              f"(width={self.interval_width:.1f}h)")
        print(f"  Input window: {self.input_win}h")

        for split in ['train', 'val', 'test']:
            n_samples = len(self.split_indices.get(split, []))
            if split in self.spells:
                spells = self.spells[split]
                n_events = sum(1 for s in spells if s.event)
                n_lines_with_events = len(set(s.line_idx for s in spells if s.event))
                durations = [s.duration for s in spells]

                print(f"\n  {split.upper()}:")
                print(f"    Timestep samples: {n_samples}")
                print(f"    Spells: {len(spells)} ({n_events} events, "
                      f"{len(spells)-n_events} censored)")
                print(f"    Lines with events: {n_lines_with_events}/{len(self.line_ids)}")
                print(f"    Duration: median={np.median(durations):.0f}h, "
                      f"mean={np.mean(durations):.0f}h, "
                      f"max={np.max(durations):.0f}h")

                # Censoring breakdown
                reasons = defaultdict(int)
                for s in spells:
                    reasons[s.censor_reason] += 1
                print(f"    Spell endings: " +
                      ", ".join(f"{r}={c}" for r, c in sorted(reasons.items())))

        print("\n" + "=" * 65)


# ============================================================================ #
#  LOSS FUNCTIONS for Discrete-Time Survival                                   #
# ============================================================================ #

class DiscreteSurvivalLoss(torch.nn.Module):
    """
    Negative log-likelihood for discrete-time survival models.

    The model outputs h_k = P(event in interval k | survived to k) for k=1..K.
    The survival function is S(k) = prod_{j=1}^{k} (1 - h_j).

    For an observation with event in interval k*:
        L = log(h_{k*}) + sum_{j=1}^{k*-1} log(1 - h_j)

    For a censored observation (survived past interval k_c):
        L = sum_{j=1}^{k_c} log(1 - h_j)

    This is equivalent to binary cross-entropy per interval, weighted by the
    at-risk mask — which is why it's so natural for neural networks.

    Reference: Gensheimer & Narasimhan (2019), "A Scalable Discrete-Time
    Survival Model for Neural Networks", PeerJ.
    """

    def __init__(self, alpha=None):
        """
        Args:
            alpha: Optional per-interval weight to emphasize early vs late events.
                   If None, all intervals weighted equally.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, hazard_pred, labels, mask):
        """
        Args:
            hazard_pred: (batch, N_lines, K) — predicted hazard probabilities
                         (after sigmoid). Must be in (0, 1).
            labels: (batch, N_lines, K) — 1 in interval of event, 0 elsewhere
            mask: (batch, N_lines, K) — 1 where subject is at risk

        Returns:
            Scalar loss (negative log-likelihood, averaged over valid observations)
        """
        eps = 1e-7
        hazard_pred = hazard_pred.clamp(eps, 1.0 - eps)

        # Log-likelihood components
        # For intervals where event occurs: log(h_k)
        # For intervals at risk but no event: log(1 - h_k)
        log_h = torch.log(hazard_pred)
        log_1_minus_h = torch.log(1.0 - hazard_pred)

        # Per-interval log-likelihood
        # Apply alpha as a pos_weight equivalent (only to the positive labels)
        if self.alpha is not None:
            alpha_tensor = self.alpha.to(labels.device)
            ll = (alpha_tensor * labels * log_h) + (1.0 - labels) * log_1_minus_h
        else:
            ll = labels * log_h + (1.0 - labels) * log_1_minus_h

        # Apply at-risk mask (only intervals where subject is alive)
        ll = ll * mask

        # Sum over intervals, mean over lines and batch
        n_valid = mask.sum().clamp(min=1.0)
        loss = -ll.sum() / n_valid

        return loss


class RankingRegularization(torch.nn.Module):
    """
    Optional concordance-based ranking loss to complement the likelihood.

    Encourages the model to correctly rank pairs of observations:
    if line A fails before line B, the model should assign higher risk to A.

    This helps when the discrete likelihood alone doesn't enforce good ranking
    (which can happen with sparse events).
    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, risk_scores, tte, event, mask):
        """
        Args:
            risk_scores: (batch, N_lines, 1) — higher = more risk
            tte: (batch, N_lines, 1)
            event: (batch, N_lines, 1)
            mask: (batch, N_lines, K) — per-interval at-risk indicator

        Returns:
            Differentiable concordance loss
        """
        # Flatten across batch and lines
        risk = risk_scores.view(-1)
        t = tte.view(-1)
        e = event.view(-1)

        # A line is considered active if it is at risk in the first interval
        # mask shape is (batch, N_lines, K). We take the first interval (index 0)
        # and flatten it to match risk, t, e.
        m = mask[:, :, 0].view(-1).bool()
        risk, t, e = risk[m], t[m], e[m]

        # Only consider pairs where one has an event
        event_idx = torch.where(e == 1)[0]
        if len(event_idx) < 2:
            return torch.tensor(0.0, device=risk.device, requires_grad=True)

        # Sample pairs for efficiency
        n_pairs = min(len(event_idx) * 10, 5000)
        idx_i = event_idx[torch.randint(len(event_idx), (n_pairs,))]
        idx_j = torch.randint(len(risk), (n_pairs,), device=risk.device)

        # Valid pairs: i had event, j survived longer (or is censored later)
        valid = t[idx_j] > t[idx_i]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=risk.device, requires_grad=True)

        idx_i, idx_j = idx_i[valid], idx_j[valid]

        # Risk of i should be HIGHER than risk of j
        diff = risk[idx_i] - risk[idx_j]
        loss = torch.mean(torch.nn.functional.softplus(-diff / self.sigma))

        return loss


class OrdinalSurvivalLoss(torch.nn.Module):
    """
    Discrete survival loss with ordinal distance penalty (Ranked Probability Score).
    
    Combines:
      1. Standard DNLL (correct likelihood for survival data)
      2. Ranked Probability Score (RPS): penalizes the cumulative distribution 
         function (CDF) to enforce ordinality. Predicting an event 3 intervals 
         away is penalized more than 1 interval away.
    """
    
    def __init__(self, alpha=None, ordinal_weight=0.5):
        """
        Args:
            alpha: per-interval event weight (from compute_interval_weights)
            ordinal_weight: weight of the ordinal penalty relative to DNLL
        """
        super().__init__()
        self.alpha = alpha
        self.ordinal_weight = ordinal_weight
    
    def forward(self, hazard_pred, labels, mask):
        eps = 1e-7
        hazard_pred = hazard_pred.clamp(eps, 1.0 - eps)
        
        log_h = torch.log(hazard_pred)
        log_1_minus_h = torch.log(1.0 - hazard_pred)
        
        # ── 1. Standard DNLL (event-weighted) ──
        if self.alpha is not None:
            alpha_tensor = self.alpha.to(labels.device)
            ll = alpha_tensor * labels * log_h + (1.0 - labels) * log_1_minus_h
        else:
            ll = labels * log_h + (1.0 - labels) * log_1_minus_h
        
        ll = ll * mask
        n_valid = mask.sum().clamp(min=1.0)
        nll = -ll.sum() / n_valid
        
        # ── 2. Ordinal Penalty via Ranked Probability Score (RPS) ──
        if self.ordinal_weight > 0:
            # Predicted CDF: F_pred(k) = 1 - S(k) = 1 - prod_{j=1}^k (1 - h_j)
            # We compute S(k) using cumsum of log(1 - h) for numerical stability
            log_S = torch.cumsum(log_1_minus_h, dim=-1)
            F_pred = 1.0 - torch.exp(log_S)
            
            # True CDF: F_true(k) = 1 for k >= true_interval, 0 otherwise
            # This is exactly the cumsum of the one-hot labels
            F_true = torch.cumsum(labels, dim=-1)
            
            # We only apply this to observations that actually have an event
            has_event = (labels.sum(dim=-1, keepdim=True) > 0.5).float()
            
            if has_event.sum() > 0:
                # Note: We compute RPS over ALL intervals (no `mask` applied), 
                # because for a known event, the true CDF is known for all K intervals.
                rps_loss = torch.mean((F_pred - F_true)**2, dim=-1) # Mean over intervals
                rps_loss = (rps_loss * has_event.squeeze(-1)).sum() / has_event.sum().clamp(min=1.0)
            else:
                rps_loss = torch.tensor(0.0, device=hazard_pred.device)
        else:
            rps_loss = torch.tensor(0.0, device=hazard_pred.device)
            
        total = nll + self.ordinal_weight * rps_loss
        
        return total

# ============================================================================ #
#  USAGE EXAMPLE                                                               #
# ============================================================================ #

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  SurvivalOutageDataManager — Usage Guide                       ║
    ╚══════════════════════════════════════════════════════════════════╝

    # ── 1. Initialize (wraps your existing OutageDataManager) ──────

    from SurvivalOutageDataManager import SurvivalOutageDataManager

    surv_manager = SurvivalOutageDataManager(
        base_manager=manager,        # Your existing OutageDataManager
        max_horizon=72,              # Predict up to 72 hours ahead
        discrete_intervals=12,       # 12 bins × 6 hours each
        train_stride=6,              # Sample every 6th hour (reduces size 6x)
        val_stride=6,
        test_stride=1,               # Dense test evaluation
    )

    # ── 2. Create DataLoaders ──────────────────────────────────────

    train_loader, val_loader, test_loader = surv_manager.create_dataloaders(
        batch_size=32
    )

    # ── 3. Inspect a sample ────────────────────────────────────────

    sample = surv_manager.construct_sample(1000)
    print(sample['line'].tte.shape)              # (N_lines, 1)
    print(sample['line'].event.shape)            # (N_lines, 1)
    print(sample['line'].discrete_labels.shape)  # (N_lines, 12)
    print(sample['line'].discrete_mask.shape)    # (N_lines, 12)

    # ── 4. Model architecture (modify your GNN+LSTM) ──────────────

    # Your existing model outputs: (batch, N_lines, 1) → sigmoid → binary prob
    # Change final layer to: (batch, N_lines, K) → sigmoid → per-interval hazard
    #
    # Pseudocode:
    #   class SurvivalGNN(YourExistingGNN):
    #       def __init__(self, ..., n_intervals=12):
    #           super().__init__(...)
    #           # Replace final binary output head:
    #           self.output_head = nn.Linear(hidden_dim, n_intervals)
    #
    #       def forward(self, data):
    #           h = self.encode(data)           # Your GNN+LSTM encoder
    #           hazard = torch.sigmoid(self.output_head(h))  # (N_lines, K)
    #           return hazard

    # ── 5. Training loop ───────────────────────────────────────────

    # loss_fn = DiscreteSurvivalLoss()
    # ranking_loss = RankingRegularization(sigma=0.1)
    #
    # for batch in train_loader:
    #     hazard = model(batch)  # (batch, N_lines, K)
    #
    #     loss = loss_fn(
    #         hazard,
    #         batch['line'].discrete_labels,
    #         batch['line'].discrete_mask,
    #     )
    #     loss += 0.1 * ranking_loss(
    #         hazard.mean(dim=-1, keepdim=True),  # Aggregate risk score
    #         batch['line'].tte,
    #         batch['line'].event,
    #         batch['line'].discrete_mask,
    #     )
    #     loss.backward()

    # ── 6. Evaluation ──────────────────────────────────────────────

    # From hazard predictions, compute:
    #   S(k) = prod_{j=1}^{k} (1 - h_j)     # Survival function
    #   P(event in next 6h) = 1 - S(1)        # For comparison with binary model
    #   C-index (concordance)                  # Primary survival metric
    #
    # Also evaluate P(event in next 6h) with PR-AUC against y_binary
    # for direct comparison with your original binary model.

    # ── 7. Classical baselines (no GNN needed) ─────────────────────

    # Export tabular data for lifelines / scikit-survival:
    # df_train = surv_manager.export_tabular('train')
    #
    # from lifelines import CoxPHFitter
    # cph = CoxPHFitter()
    # cph.fit(df_train, duration_col='duration', event_col='event')
    #
    # from sksurv.ensemble import RandomSurvivalForest
    # rsf = RandomSurvivalForest(n_estimators=100)
    # # Format for sksurv: y = structured array of (event, duration)
    """)
