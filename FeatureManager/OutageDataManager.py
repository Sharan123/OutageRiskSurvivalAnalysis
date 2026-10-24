import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import h5py
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from torch_geometric.data import HeteroData
import hashlib
from shapely.ops import nearest_points
from shapely.geometry import Point
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, Sampler
from sklearn.preprocessing import MinMaxScaler

# Add base directory to path if needed
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from FeatureManager.atomic_io import atomic_path
from BPA_Spatial_Preprocessing.BPAGraphCreator import BPAGraphCreator
from BPA_Line_Outage_Matcher.BPA_Outage_Line_Matcher import BPAOutageMatcher
from WeatherPreprocessor.WeatherPreprocessor import WeatherPreprocessor
from FeatureManager.outage_events import (build_event_registry, matrices_from_registry, DEFAULT_WEATHER_CAUSES,
                                          OUTAGE_EVENTS_VERSION, file_fingerprint)

# v2 (2026-10-08): edges sorted by (dst, src). BPAGraphCreator returns Python sets whose
# iteration order changes between processes, so edge ORDER (not the edge set) differed
# between cache builds/machines; sorting makes every machine build identical inputs.
GRAPH_BUILD_VERSION = "graph_sorted_edges_v2"


def _sorted_edges(index: torch.Tensor, attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if index.numel() == 0:
        return index, attr
    order = np.lexsort((index[0].numpy(), index[1].numpy()))  # by dst, then src
    order = torch.from_numpy(order)
    return index[:, order].contiguous(), attr[order].contiguous()

class OutageDataset(Dataset):
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


class StratifiedTrainSampler(Sampler):
    def __init__(self, manager, batch_size, negative_ratio=None):
        self.manager = manager
        self.batch_size = int(batch_size)
        self.negative_ratio = negative_ratio

    def __iter__(self):
        self.manager.resample_train_indices(negative_ratio=self.negative_ratio)
        train_indices = self.manager.split_indices.get('train', [])
        if not train_indices:
            return iter([])

        pos_set = self.manager.train_positive_set
        if not pos_set and self.manager.train_positive_indices:
            pos_set = set(self.manager.train_positive_indices)

        pos_positions = [i for i, idx in enumerate(train_indices) if idx in pos_set]
        neg_positions = [i for i, idx in enumerate(train_indices) if idx not in pos_set]

        order = self._build_order(pos_positions, neg_positions)
        return iter(order)

    def __len__(self):
        return len(self.manager.split_indices.get('train', []))

    def _build_order(self, pos_positions, neg_positions):
        total = len(pos_positions) + len(neg_positions)
        if total == 0:
            return []

        if self.batch_size < 2 or len(pos_positions) < 2:
            all_positions = pos_positions + neg_positions
            np.random.shuffle(all_positions)
            return all_positions

        num_full_batches = total // self.batch_size
        remainder = total % self.batch_size
        batch_sizes = [self.batch_size] * num_full_batches
        if remainder:
            batch_sizes.append(remainder)

        required_pos = sum(2 for size in batch_sizes if size >= 2)
        if len(pos_positions) < required_pos:
            all_positions = pos_positions + neg_positions
            np.random.shuffle(all_positions)
            return all_positions

        np.random.shuffle(pos_positions)
        np.random.shuffle(neg_positions)

        batches = [[] for _ in batch_sizes]
        pos_idx = 0
        for i, size in enumerate(batch_sizes):
            need = 2 if size >= 2 else 0
            if need:
                batches[i].extend(pos_positions[pos_idx:pos_idx + need])
                pos_idx += need

        remaining = pos_positions[pos_idx:] + neg_positions
        np.random.shuffle(remaining)

        remaining_idx = 0
        for i, size in enumerate(batch_sizes):
            capacity = size - len(batches[i])
            if capacity > 0:
                batches[i].extend(remaining[remaining_idx:remaining_idx + capacity])
                remaining_idx += capacity

        for batch in batches:
            np.random.shuffle(batch)
        np.random.shuffle(batches)

        order = [idx for batch in batches for idx in batch]
        return order


def create_dataloaders(manager, batch_size=32, *, stratified_train=True, negative_ratio=None, num_workers=4, pin_memory=True, prefetch_factor=2):
    """
    Creates DataLoaders for train, validation, and test sets.
    Train loader uses dynamic negative sampling and soft-stratified ordering by default.
    """
    train_ds = manager.get_dataset('train')
    val_ds = manager.get_dataset('val')
    test_ds = manager.get_dataset('test')

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0 and prefetch_factor is not None:
         loader_args["prefetch_factor"] = prefetch_factor

    if stratified_train:
        train_sampler = StratifiedTrainSampler(
            manager=manager,
            batch_size=batch_size,
            negative_ratio=negative_ratio,
        )
        train_loader = DataLoader(train_ds, shuffle=False, sampler=train_sampler, **loader_args) 
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_args)

    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader


class OutageDataManager:
    def __init__(self, 
                 graph_creator: BPAGraphCreator,
                 weather_preprocessor: WeatherPreprocessor,
                 outage_matcher: BPAOutageMatcher,
                 start_date="2016-01-01", 
                 end_date="2023-12-31",
                 train_split="2022-01-01",
                 val_split="2023-01-01",
                 input_window_hours=12,
                 output_window_hours=6,
                 tiny_mode=False,
                 processed_dir=None,
                 verbose=True,
                 negative_ratio=3,
                 valid_stride=1
                 ):
        
        self.graph_creator = graph_creator
        self.weather_preprocessor = weather_preprocessor
        self.outage_matcher = outage_matcher
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.train_split = pd.Timestamp(train_split)
        self.val_split = pd.Timestamp(val_split)
        self.input_win = input_window_hours
        self.output_win = output_window_hours
        self.tiny_mode = tiny_mode
        self.verbose = verbose
        
        # Resolve splits from WeatherPreprocessor if available to ensure config hash matches actual data used
        self.train_end_idx = None
        self.val_end_idx = None
        
        try:
            split_bounds = self.weather_preprocessor.get_split_boundaries()
            self.train_end_idx = split_bounds['train'][1]
            self.val_end_idx = split_bounds['valid'][1]
            if self.verbose:
                print(f"Aligned with WeatherPreprocessor splits: Train End={self.train_end_idx}, Val End={self.val_end_idx}")
        except Exception as e:
            if self.verbose:
                print(f"Using date-based splits (WeatherPreprocessor splits unavailable: {e})")
            # Calculate indices from dates
            self.train_end_idx = int((self.train_split - self.start_date).total_seconds() / 3600)
            self.val_end_idx = int((self.val_split - self.start_date).total_seconds() / 3600)

        # File paths
        self.processed_dir = Path(processed_dir) if processed_dir else Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration Hash for caching
        # Use the resolved indices in the hash instead of the input date strings
        # The key also covers the outage data itself (2026-10-08): a fingerprint of the
        # preprocessed outage CSV, the weather-cause list and the outage-event rules.
        # Previously a regenerated CSV would have silently reused stale matrices.
        csv_path = getattr(self.outage_matcher, "csv_path", None) or (base_dir / "BPA_Line_Outage_Matcher/preprocessed_outages_2018_2023.csv")
        self.outage_csv_fingerprint = file_fingerprint(csv_path)[:16]
        config_str = (f"{start_date}_{end_date}_{self.train_end_idx}_{self.val_end_idx}_{input_window_hours}_{output_window_hours}"
                      f"_{tiny_mode}_{negative_ratio}_{valid_stride}_{self.outage_csv_fingerprint}_{OUTAGE_EVENTS_VERSION}_{GRAPH_BUILD_VERSION}"
                      f"_{'|'.join(DEFAULT_WEATHER_CAUSES)}")
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()
        self.meta_path = self.processed_dir / f"dataset_meta_{self.config_hash}.pkl"

        # Attributes to be populated
        self.line_features_static = None  # Tensor (N_lines, F_static)
        self.line_ids = None              # List[str] of XRefCd values aligned to line node indices
        self.line_xref_idx = None         # LongTensor (N_lines,) used for XRefCd embeddings in the model
        self.weather_node_map = {}        # (lat, lon) -> index (inside the geometric node tensor defined in `get_weather_tensor_for_window`)
        self.weather_edges_index = None   # Tensor (2, Num_Edges) for Weather->Line # row 0: Weather (src), row 1: Line (dst)
        self.weather_edges_attr = None    # Tensor (Num_Edges, 2) for (Delta Lat, Delta Lon)
        self.edge_attr_scaler = None      # Scaler for edge attributes
        self.line_feature_scaler = None   # Scaler for static line features
        
        # Topology
        self.line_edges_index = None      # Tensor (2, Num_Line_Edges) for Line<->Line
        self.line_edges_attr = None       # Tensor (Num_Line_Edges, 2) -> [Is_Direct, Is_Substation]

        self.outage_matrix = None         # Matrix (Time x Lines) for weather-related outages (labels)
        self.all_outages_matrix = None    # Matrix (Time x Lines) for all outages (features/masks)
        self.event_registry = None        # DataFrame of normalised outage records (§3.4); None for old caches
        self.split_indices = {}           # Dictionary {'train': [], 'val': [], 'test': []}
        self.train_positive_indices = []
        self.train_positive_set = set()
        self.train_negative_candidates = []
        self.train_negative_indices = []
        self.train_negative_ratio = None
        self.train_sampling_end_idx = None
        self.val_sampling_end_idx = None
        self.max_valid_idx = None
        self.unique_weather_cells = []    # List of (lat, lon) tuples
        
        # Initialize
        self._initialize_dataset(negative_ratio=negative_ratio, valid_stride=valid_stride)
        self._precompute_time_since_last()

    def _precompute_time_since_last(self):
        """
        Creates a matrix of shape (Total_Hours, Num_Lines)
        Value = Normalized time since last outage (0.0 to 1.0)
        Based on all outage causes to reflect resiliency.
        """
        print("Pre-computing time-since-last-outage features...")
        num_timesteps, num_lines = self.all_outages_matrix.shape
        
        # Initialize with a large number (e.g., 1 year in hours) representing "Infinity/Never"
        # We normalize by 2160 hours (3 months/90 days) as per your previous code
        max_hours = 2160.0 
        
        # This will hold the raw counter
        time_since = np.full((num_timesteps, num_lines), 1.0, dtype=np.float32)
        
        counter = np.full(num_lines, max_hours)
        
        for t in range(num_timesteps):
            # Where outage is occurring NOW, reset counter to 0
            is_outage = (self.all_outages_matrix[t] == 1)
            counter[is_outage] = 0.0
            
            # Increment everyone else
            counter[~is_outage] += 1.0
            
            # Store (clipped to max)
            clipped_counter = np.minimum(counter, max_hours)
            time_since[t] = clipped_counter / max_hours
        
        self.time_since_matrix = time_since

    def _initialize_dataset(self, negative_ratio=3, valid_stride=1):
        """Main orchestration method to load or build the dataset metadata."""
        rebuild = False
        if self.meta_path.exists():
            if self.verbose: print(f"Loading cached dataset metadata from {self.meta_path}...")
            with open(self.meta_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check for new fields
            if (
                'all_outages_matrix' not in data or data.get('event_registry') is None
            ):
                print("Cached dataset missing required metadata (scalers/line_ids/outages). Rebuilding...")
                rebuild = True
            else:
                for k, v in data.items():
                    setattr(self, k, v)
        else:
            rebuild = True

        if rebuild:
            if self.verbose: print("Building dataset from source...")
            self._build_static_graph_structure()
            self._process_outages_and_timeline()

        self._generate_sampling_indices(negative_ratio=negative_ratio, valid_stride=valid_stride)

        if rebuild:
            # Cache the metadata
            save_dict = {
                'line_features_static': self.line_features_static,
                'line_ids': self.line_ids,
                'line_xref_idx': self.line_xref_idx,
                'weather_node_map': self.weather_node_map,
                'weather_edges_index': self.weather_edges_index,
                'weather_edges_attr': self.weather_edges_attr,
                'edge_attr_scaler': self.edge_attr_scaler,
                'line_feature_scaler': self.line_feature_scaler,
                'line_edges_index': self.line_edges_index,
                'line_edges_attr': self.line_edges_attr,
                'outage_matrix': self.outage_matrix,
                'all_outages_matrix': self.all_outages_matrix,
                'split_indices': self.split_indices,
                'unique_weather_cells': self.unique_weather_cells,
                'event_registry': self.event_registry,
            }
            with atomic_path(self.meta_path) as tmp, open(tmp, 'wb') as f:  # never leave a half-written cache
                pickle.dump(save_dict, f)

    def _build_static_graph_structure(self):
        """
        1. Identify unique weather cells required by the buffer zones.
        2. Create Bipartite Edges (Weather -> Line) and calculate offsets.
        3. Create Line Topology Edges (Line <-> Line).
        4. Extract and Scale static line features.
        """
        # --- 1. Weather Grid Construction ---
        # Get all unique cells used by the graph creator's buffer zones
        all_cells = set()
        # BPAGraphCreator should have popualted cell_coverage_cache after run
        if not self.graph_creator.cell_coverage_cache:
            if self.verbose: print("GraphCreator cache is empty. Running graph.get_cells_coverage() first.")
            self.graph_creator.get_cells_coverage()

        for xrefcd, cells in self.graph_creator.cell_coverage_cache.items():
            all_cells.update(cells)
        
        # Sort for deterministic indexing
        self.unique_weather_cells = sorted(list(all_cells))
        self.weather_node_map = {cell: i for i, cell in enumerate(self.unique_weather_cells)}
        
        if self.verbose: print(f"Total Unique Weather Nodes (Bipartite Left Side): {len(self.unique_weather_cells)}")

        # --- 2. Bipartite Edge Construction (Weather -> Line) ---
        src_nodes = [] # Weather indices
        dst_nodes = [] # Line indices
        edge_attrs = [] # (Delta Lat, Delta Lon)

        # Get ordered list of lines to ensure index alignment
        # Use the graph creator as the source of truth for nodes so the topology stays consistent
        self.line_ids = list(self.graph_creator.cell_coverage_cache.keys())
        line_id_to_idx = {xref: i for i, xref in enumerate(self.line_ids)}
        self.line_xref_idx = torch.arange(len(self.line_ids), dtype=torch.long)

        for xrefcd, cells in tqdm(self.graph_creator.cell_coverage_cache.items(), desc="Building Bipartite Edges"):
            line_idx = line_id_to_idx[xrefcd]
            
            # Get the Line's "Center" or representative point for offset calculation
            # Note: A line is a shape, but we need a vector offset. 
            # Strategy: Calculate offset from Weather Cell Center to the CLOSEST point on the line.
            line_geom = self.graph_creator.lines[self.graph_creator.lines['XRefCd'] == xrefcd].geometry.iloc[0]
            
            for cell in cells:
                weather_idx = self.weather_node_map[cell]
                
                # Find nearest point on line to this weather cell
                # Using shapely ops cell is (lon, lat)
                p_cell = Point(cell[0], cell[1])
                p_line_nearest = nearest_points(p_cell, line_geom)[1]
                
                # Delta: Vector FROM Weather TO Line
                d_lon = p_line_nearest.x - p_cell.x
                d_lat = p_line_nearest.y - p_cell.y
                
                src_nodes.append(weather_idx)
                dst_nodes.append(line_idx)
                edge_attrs.append([d_lat, d_lon])

        self.weather_edges_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long) # Shape (2, Num_Edges) # Need torch.long for PyG, otherwise index error # src=Weather, dst=Line
        
        # Scale Edge Attributes (Delta Lat, Delta Lon) to [-1, 1]
        edge_attrs_np = np.array(edge_attrs, dtype=np.float32)
        self.edge_attr_scaler = MinMaxScaler(feature_range=(-1, 1))
        edge_attrs_scaled = self.edge_attr_scaler.fit_transform(edge_attrs_np)
        
        self.weather_edges_attr = torch.tensor(edge_attrs_scaled, dtype=torch.float) # Shape (Num_Edges, 2)
        self.weather_edges_index, self.weather_edges_attr = _sorted_edges(self.weather_edges_index, self.weather_edges_attr)

        # --- 3. Line Topology (Line <-> Line) ---
        # Extract from line_ends_dict in GraphCreator
        # This needs to be mapped to our 0..N indices
        l_src = []
        l_dst = []
        l_attr = [] # [Is_Direct, Is_Substation]
        
        for i, xref_a in enumerate(self.line_ids):
            # Calls the new method in BPAGraphCreator that returns a dict

            connections = self.graph_creator.get_connected_lines(xref_a)

            # Direct Connections (Physical touch) -> Type [1, 0] 
            for xref_b in connections['direct']:
                if xref_b in line_id_to_idx:
                    l_src.append(i)
                    l_dst.append(line_id_to_idx[xref_b])
                    l_attr.append([1.0, 0.0])
            
            # Substation Connections (Shared Bus) -> Type [0, 1]
            for xref_b in connections['substation']:
                if xref_b in line_id_to_idx:
                    l_src.append(i)
                    l_dst.append(line_id_to_idx[xref_b])
                    l_attr.append([0.0, 1.0])
        
        self.line_edges_index = torch.tensor([l_src, l_dst], dtype=torch.long)
        self.line_edges_attr = torch.tensor(l_attr, dtype=torch.float)
        self.line_edges_index, self.line_edges_attr = _sorted_edges(self.line_edges_index, self.line_edges_attr)

        # --- 4. Static Line Features ---
        # Get Dictionary: {xrefcd: [Voltage, Length, L, T, TT]}
        feats_dict = self.outage_matcher.get_line_features(self.line_ids) 
        # Ensure it returns numerical cols: [Voltage, Length, Type_L, Type_T, Type_TT]
        # Convert to Matrix aligned with self.line_ids
        feats_list = []
        for xref in self.line_ids:
            if xref not in feats_dict:
                raise ValueError(f"Line XRefCd {xref} missing in features dictionary.")
            feats_list.append(feats_dict[xref])
        
        feats_array = np.array(feats_list, dtype=np.float32)
        self.line_feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        feats_array[:, 0:2] = self.line_feature_scaler.fit_transform(feats_array[:, 0:2])
        self.line_features_static = torch.tensor(feats_array, dtype=torch.float)
        if self.verbose: print(f"Static Features Scaled. Shape: {self.line_features_static.shape}")

    def _process_outages_and_timeline(self):
        """
        Creates master timeline matrices: [Time_Steps, Num_Lines]
        Values: 0 = Active/Healthy, 1 = Outage Event
        """
        # 1. Load Data using Matcher's Logic
        if self.verbose: print(f"Loading outages from the default path...")
        self.outage_matcher.load_csv()
        outages = self.outage_matcher.grouped

        # Extend end_date to include the full last day (up to 23:00)
        effective_end = self.end_date + pd.Timedelta(hours=23)
        full_range = pd.date_range(start=self.start_date, end=effective_end, freq='h')

        print("Mapping outages to timeline...")

        # IMPLEMENTATION_STEPS §3.4: the rounding/skip rules now live in
        # FeatureManager/outage_events.py and produce a normalised registry that
        # keeps Outage ID, cause and per-line episode boundaries. The binary
        # matrices are rebuilt from that registry, so both share one code path
        # and the matrices are unchanged from the previous inline loop.
        self.event_registry = build_event_registry(
            outages, self.line_ids, start_date=self.start_date, end_date=self.end_date,
            weather_causes=DEFAULT_WEATHER_CAUSES,
        )
        skipped = self.event_registry[self.event_registry['mapping_status'] != 'mapped']
        if self.verbose and len(skipped):
            print(f"Outage records not mapped to the timeline: {skipped['mapping_status'].value_counts().to_dict()}")
        self.outage_matrix, self.all_outages_matrix = matrices_from_registry(
            self.event_registry, len(full_range), len(self.line_ids)
        )

    # TODO: Use WeightedRandomSampler or Focal Loss if you increase the data size
    # TODO: Using WeatherProcessor's transform_ds, generate separate samples for train/val/test in the same manner as in the WeatherPreprocessor
    def _prepare_train_sampling_pool(self, train_end_idx):
        """
        Precomputes positives and negative candidates for dynamic train sampling.
        """
        print("Generating Train Samples (Anchor)...")
        positive_samples = set()

        diff_matrix = np.diff(self.outage_matrix, axis=0, prepend=0)
        rows, cols = np.where(diff_matrix == 1)

        for t_outage, _ in zip(rows, cols):
            earliest_start = t_outage - self.input_win - self.output_win + 1
            latest_start = t_outage - self.input_win

            start = max(0, earliest_start)
            end = min(latest_start, train_end_idx - 1)

            if start <= end:
                positive_samples.update(range(start, end + 1))

        self.train_positive_indices = sorted(positive_samples)
        self.train_positive_set = set(self.train_positive_indices)

        all_train_indices = np.arange(0, train_end_idx)
        if len(all_train_indices) == 0:
            self.train_negative_candidates = []
            return

        valid_mask = np.ones(len(all_train_indices), dtype=bool)
        pos_in_range = [p for p in self.train_positive_indices if p < train_end_idx]
        if pos_in_range:
            valid_mask[pos_in_range] = False

        self.train_negative_candidates = all_train_indices[valid_mask].tolist()

    def resample_train_indices(self, negative_ratio=None):
        """
        Resamples train negatives while keeping all positives.
        """
        if self.train_sampling_end_idx is None:
            raise RuntimeError("Train sampling pool not initialized; call _generate_sampling_indices first.")

        if negative_ratio is not None:
            self.train_negative_ratio = negative_ratio

        ratio = self.train_negative_ratio if self.train_negative_ratio is not None else 3
        num_neg = int(len(self.train_positive_indices) * ratio)

        if num_neg <= 0:
            self.train_negative_indices = []
        else:
            candidates = np.array(self.train_negative_candidates)
            if len(candidates) > num_neg:
                self.train_negative_indices = np.random.choice(candidates, num_neg, replace=False).tolist()
            else:
                self.train_negative_indices = candidates.tolist()

        self.split_indices['train'] = sorted(self.train_positive_indices + self.train_negative_indices)
        if self.tiny_mode:
            self.split_indices['train'] = self.split_indices['train'][:100]

        return self.split_indices['train']

    def _generate_sampling_indices(self, negative_ratio=3, valid_stride=1):
        """
        Generates sampling indices for train, validation, and test sets.
        Ensures that time windows (input+output) do not cross split boundaries.
        """
        total_timesteps = self.outage_matrix.shape[0]
        max_valid_idx = total_timesteps - self.input_win - self.output_win
        
        self.max_valid_idx = max_valid_idx
        
        # Calculate Safe Sampling Stop Indices (Exclusive)
        # We need: t + input + output <= Boundary
        # So: t <= Boundary - input - output
        # Stop Index = Boundary - input - output + 1 (since range is exclusive)
        
        window_len = self.input_win + self.output_win
        
        # Train Sampling Stop
        # Ideally, train samples shouldn't peek into validation period (starting at self.train_end_idx)
        train_sampling_stop = self.train_end_idx - window_len
        # Clamp to ensure we don't go negative or exceed data
        train_sampling_stop = max(0, min(train_sampling_stop, max_valid_idx) + 1)
        
        # Validation Sampling Stop
        # Val samples shouldn't peek into Test period (starting at self.val_end_idx)
        val_sampling_stop = self.val_end_idx - window_len
        val_sampling_stop = max(train_sampling_stop, min(val_sampling_stop, max_valid_idx) + 1)
        
        self.train_sampling_end_idx = train_sampling_stop
        self.val_sampling_end_idx = val_sampling_stop

        print(f"Split Indices (Sample Start Ranges): Train [0, {train_sampling_stop}), Val [{self.train_end_idx}, {val_sampling_stop}), Test [{self.val_end_idx}, {max_valid_idx + 1})")

        # --- Train Set (Dynamic Negatives) ---
        self._prepare_train_sampling_pool(train_sampling_stop)
        self.train_negative_ratio = negative_ratio
        self.resample_train_indices(negative_ratio=negative_ratio)
        
        # --- Validation Set (Dense) ---
        print("Generating Validation Samples (Dense)...")
        # Start validation samples at train_end_idx to ensure they are strictly in validation period (mostly)
        # Assuming gap between train_sampling_stop and train_end_idx is handled by the fact that those windows would overlap boundary.
        val_start = self.train_end_idx
        val_end = val_sampling_stop
        
        if val_start < val_end:
            self.split_indices['val'] = list(range(val_start, val_end, valid_stride))
        else:
            self.split_indices['val'] = []
            
        # --- Test Set (Dense) ---
        print("Generating Test Samples (Dense)...")
        # Test starts at val_end_idx
        test_start = self.val_end_idx
        test_end = max_valid_idx + 1 # Exclusive
        
        if test_start < test_end:
            self.split_indices['test'] = list(range(test_start, test_end, 1)) 
        else:
            self.split_indices['test'] = []
            
        if self.tiny_mode:
            print("Tiny Mode: Truncating datasets...")
            self.split_indices['train'] = self.split_indices['train'][:100]
            self.split_indices['val'] = self.split_indices['val'][:50]
            self.split_indices['test'] = self.split_indices['test'][:50]
            
        print(f"Dataset Prepared: Train={len(self.split_indices['train'])}, Val={len(self.split_indices['val'])}, Test={len(self.split_indices['test'])}")

    def get_weather_tensor_for_window(self, time_idx, window_size):
        """
        Fetches raw weather features for the specified time window.
        Uses the pre-calculated unique_weather_cells to slice the HDF5 efficiently (if possible)
        or uses the WeatherPreprocessor to fetch specific cells.
        """
        # Returns Tensor: (T_win, N_cells, F_weather)
        # CAUTION: This is the heaviest part. Ideally, HDF5 should be structured [Time, Cells, Feats]
        # and we just slice [t:t+w, cell_indices, :]
        
        # Pseudocode for optimized fetch
        features = self.weather_preprocessor.fetch_vectorized(
            global_time_idx=time_idx,
            window_size=window_size
        )
        # Note: Permutation to (Nodes, Time, Feats) happens in construct_sample
        return torch.from_numpy(features).float()

    def get_dataset(self, mode='train'):
        """
        Returns a PyTorch Dataset for the specified mode.
        """
        if mode not in self.split_indices:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.split_indices.keys())}")
        
        return OutageDataset(self, mode)

    def construct_sample(self, t_start):
        """
        Constructs the HeteroData object for a specific time window starting at t_start.
        """
        t_predict = t_start + self.input_win
        t_end = t_predict + self.output_win
        
        data = HeteroData()

        # Explicitly set num_nodes to avoid inference warnings/errors
        data['line'].num_nodes = len(self.line_ids)

        data['line'].x_static = self.line_features_static # (N_lines, F_static); F_static=5
        data['line'].xref_idx = self.line_xref_idx # (N_lines,)
        status_hist = self.outage_matrix[t_start:t_predict, :] # (T_in, N_lines)
        
        time_since_hist = self.time_since_matrix[t_start:t_predict, :] # (T_in, N_lines)
        
        # Stack them: 
        # Feature 0: Is Line Active? (0/1)
        # Feature 1: Time Since Last (0.0-1.0)
        
        combined_dynamic = np.stack([status_hist, time_since_hist], axis=-1) # (T_in, N_lines, 2)
        
        # Transpose for PyG: (N_lines, T_in, Features)
        data['line'].x_dynamic = torch.from_numpy(combined_dynamic).float().permute(1, 0, 2)
        
        # --- 2. Node Features: Weather ---
        # Shape: (T_in, N_weather, F_weather)
        # We load T_in steps of weather
        weather_feats = self.get_weather_tensor_for_window(t_start, self.input_win)
        # Transpose to (N_weather, T_in, F) if needed, depends on fetcher
        data['weather'].x = weather_feats.permute(1, 0, 2)
        data['weather'].num_nodes = data['weather'].x.size(0)
        
        # --- 3. Edges ---
        # These are pre-calculated and static for the window
        data['weather', 'affects', 'line'].edge_index = self.weather_edges_index # (2, N_edges)
        data['weather', 'affects', 'line'].edge_attr = self.weather_edges_attr # (N_edges, 2) -> DeltaLat, DeltaLon
        
        data['line', 'connected_to', 'line'].edge_index = self.line_edges_index # (2, N_line_edges)
        data['line', 'connected_to', 'line'].edge_attr = self.line_edges_attr # (N_line_edges, 2) -> Is_Direct, Is_Substation
        
        # --- 4. Labels & Masks (The Critical Phase 1 Logic) ---
        
        # Target: Is there an outage in [t_predict, t_end)?
        # Shape: (N_lines, 1)
        future_status = self.outage_matrix[t_predict:t_end, :] # (T_out, N_lines)
        # If ANY 1 in the window -> Label 1
        y = np.max(future_status, axis=0)
        data['line'].y = torch.from_numpy(y).float().unsqueeze(-1) # (N_lines, 1)
        
        # Mask: Is the line ALIVE at t_predict?
        # Check status at t_predict - 1 (the last moment of input)
        # If it was 1 (outage) at the end of input, it is DEAD for prediction.
        # Status 0 = Alive, 1 = Outage
        last_known_status = self.all_outages_matrix[t_predict - 1, :]
        
        # is_active = 1 if status was 0 (Healthy)
        is_active = 1 - last_known_status

        # Mask out lines with non-weather outages in the prediction window.
        future_all = self.all_outages_matrix[t_predict:t_end, :]
        future_weather = self.outage_matrix[t_predict:t_end, :]
        non_weather_future = (future_all == 1) & (future_weather == 0)
        has_non_weather_future = np.any(non_weather_future, axis=0)
        is_active = is_active & (~has_non_weather_future)
        data['line'].is_active = torch.from_numpy(is_active).bool().unsqueeze(-1) # (N_lines, 1)
        
        return data

# Potential changes if slow
# Proposed changes (no edits yet):
                                                                                                                                                                                                                                                                                                                                   
#   - Keep HDF5 safe while reducing overhead by opening one file per worker and reusing it:                                                                                                                                                                                                                                          
#       - Add a worker_init_fn that opens the HDF5 file and stores the handle on the dataset or WeatherPreprocessor instance.                                                                                                                                                                                                        
#       - Use persistent_workers=True so workers (and their file handles) stay alive across epochs.                                                                                                                                                                                                                                  
#   - Consider a multiprocessing_context="spawn" on Linux if you ever move to forking (HDF5 + fork can be problematic).  