"""
SurvivalOutageDataManager
=========================

Reformulates power-line outage prediction as cause-specific survival analysis
over *operating spells*:

    "What is the weather outage hazard while the line is operating and observable?"

Key concepts (IMPLEMENTATION_STEPS.md §3):

    Operating spell   A maximal run of hours in which the line is in no outage
                      at all (`all_outages_matrix == 0`). Recovery from any outage
                      starts a new spell. Spells are physical (they are not cut
                      at train/val/test boundaries); each split sees an
                      observation *segment* of the spell.

    Eligibility       A (prediction_time, line) row is in the survival population
                      iff the line is operating at the prediction time
                      (`all_outages_matrix[t_predict, line] == 0`). Lines in outage
                      stay in the graph for message passing but carry
                      eligible=False and contribute nothing to any loss/metric.

    Termination       The observation for a row ends at the earliest of
                      weather onset (event), non-weather onset (censoring),
                      split boundary, end of data, or the horizon cap H.
                      Raw termination (time + reason) is kept next to the
                      horizon-capped (tte, event) target.

    Event linkage     Each spell keeps the registry index of the outage record
                      that terminated it, so a label can be traced to a source
                      `Outage ID` / line episode (§3.4). Ambiguous candidates are
                      never silently resolved; they are listed in an audit table.

Everything is derived from ONE spell table, so the time-to-event labels and the
spell summary can no longer disagree about precedence or boundaries (§3.3).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from FeatureManager.atomic_io import atomic_path
from FeatureManager.outage_events import MAPPED, build_event_registry
from SurvivalModels.contract import (
    ADMINISTRATIVE_TERMINATIONS,
    EXCL_ALREADY_OUTAGED,
    EXCL_NONE,
    LABEL_CONTRACT_VERSION,
    TERM_END_OF_DATA,
    TERM_INELIGIBLE,
    TERM_MAX_H,
    TERM_NON_WEATHER,
    TERM_SPLIT,
    TERM_WEATHER,
    TERMINATION_REASONS,
    WEATHER_CAUSES,
    discrete_targets,
    horizon_outcomes,
    interval_bounds,
)

SPLITS = ("train", "val", "test")
SPLIT_CODES = {name: i for i, name in enumerate(SPLITS)}

# Cache entries that a v1 label cache must contain; anything older is rebuilt
# (§3.5: TTE/event presence alone is not sufficient).
_REQUIRED_CACHE_KEYS = {
    "label_contract_version", "spell_table", "split_indices", "event_registry",
    "spell_audit", "source_fingerprint",
}

# Spell-termination ID resolution status
ID_UNIQUE = "unique"
ID_AMBIGUOUS = "ambiguous_multiple_candidates"
ID_NO_CANDIDATE = "no_candidate_record"
ID_NOT_APPLICABLE = "not_applicable"


@dataclass
class Spell:
    """Legacy spell record. Kept only so pre-v1 caches (survival_meta_*.pkl) can
    still be unpickled for audit (§3.5); the v1 labels use `spell_table`."""
    line_idx: int
    t_start: int
    t_end: int
    event: bool
    duration: int
    censor_reason: str


# ============================================================================ #
#  Survival Dataset                                                            #
# ============================================================================ #

class SurvivalDataset(Dataset):
    """Yields one survival-formatted HeteroData snapshot per scheduled prediction time."""

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
    Wraps an OutageDataManager and adds operating-spell survival targets.

    Args:
        base_manager: initialised OutageDataManager (or a fixture exposing the
            same attributes).
        max_horizon: horizon cap H in hours. Persisted in the model config and
            checkpoints; never inferred from observed follow-up (§2.1).
        discrete_intervals: number of equal-width hazard intervals K.
        train_stride / val_stride / test_stride: sampling cadence per split.
        weather_adjacency: 'none' (default), 'grid8' or 'grid4'. Adds a
            ('weather','near','weather') graph for the weather-message pilot (§9.5):
            each 0.1-degree cell is linked to the EXISTING cells among its 8 (or 4)
            grid neighbours. Input-only; labels unaffected.
        event_registry: optional pre-built registry (fixtures); otherwise taken
            from the base manager or rebuilt from its outage matcher.
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
        weather_adjacency: str = "none",
        event_registry: Optional[pd.DataFrame] = None,
    ):
        self.base = base_manager
        self.max_horizon = int(max_horizon)
        self.discrete_intervals = int(discrete_intervals)
        self.train_stride = int(train_stride)
        self.val_stride = int(val_stride)
        self.test_stride = int(test_stride)
        self.verbose = verbose
        if weather_adjacency not in ("none", "grid8", "grid4"):
            raise ValueError(f"weather_adjacency must be 'none', 'grid8' or 'grid4', got {weather_adjacency!r}")
        self.weather_adjacency = weather_adjacency

        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        # Fixed grid built once from the configured horizon (§5.1).
        self.interval_width = self.max_horizon / self.discrete_intervals
        self.interval_bounds = interval_bounds(self.max_horizon, self.discrete_intervals)

        self._forward_base_attributes()
        self.total_t = int(self.all_outages_matrix.shape[0])
        self.n_lines = int(self.all_outages_matrix.shape[1])
        self.split_bounds = {
            "train": (0, int(self.train_end_idx)),
            "val": (int(self.train_end_idx), int(self.val_end_idx)),
            "test": (int(self.val_end_idx), self.total_t),
        }

        self._supplied_registry = event_registry
        self.source_fingerprint = self._compute_source_fingerprint()

        # Cache key includes the label-contract version, interval convention,
        # cause mapping and source fingerprint (§3.5), so an older cache can
        # never be reused as though it had the new metadata.
        config_str = "_".join(
            str(x) for x in (
                getattr(self.base, "config_hash", "nohash"), self.max_horizon, self.discrete_intervals,
                self.train_stride, self.val_stride, self.test_stride, LABEL_CONTRACT_VERSION,
                "|".join(WEATHER_CAUSES), "half_open", self.source_fingerprint,
            )
        )
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()
        self.meta_path = os.path.join(self.processed_dir, f"survival_labels_{self.config_hash}.pkl")

        self.weather_weather_edge_index = None
        self.weather_weather_edge_attr = None
        if self.weather_adjacency != "none":
            self._build_weather_grid_edges(self.weather_adjacency)

        self._initialize_dataset()

        if self.verbose:
            self._print_summary()

    # ------------------------------------------------------------------ #
    #  Setup                                                              #
    # ------------------------------------------------------------------ #

    def _forward_base_attributes(self):
        """Forward commonly accessed attributes from the base manager."""
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
        self.weather_preprocessor = getattr(self.base, "weather_preprocessor", None)
        self.start_date = pd.Timestamp(getattr(self.base, "start_date", "2016-01-01"))

    def _compute_source_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.outage_matrix).tobytes())
        h.update(np.ascontiguousarray(self.all_outages_matrix).tobytes())
        h.update("|".join(map(str, self.line_ids)).encode())
        h.update(f"{self.train_end_idx}_{self.val_end_idx}_{self.input_win}".encode())
        return h.hexdigest()[:16]

    def _get_event_registry(self) -> pd.DataFrame:
        if self._supplied_registry is not None:
            return self._supplied_registry.copy()
        reg = getattr(self.base, "event_registry", None)
        if reg is not None:
            return reg.copy()
        matcher = getattr(self.base, "outage_matcher", None)
        if matcher is None:
            raise RuntimeError(
                "No event registry available: the base manager has no `event_registry` and no "
                "`outage_matcher` to rebuild it from. Source outage IDs are required by §3.4."
            )
        if getattr(matcher, "grouped", None) is None:
            matcher.load_csv()
        return build_event_registry(
            matcher.grouped, self.line_ids,
            start_date=getattr(self.base, "start_date"), end_date=getattr(self.base, "end_date"),
            weather_causes=WEATHER_CAUSES,
        )

    def _initialize_dataset(self):
        """Load the versioned label cache or rebuild it."""
        rebuild = True
        if os.path.exists(self.meta_path):
            if self.verbose:
                print(f"Loading cached survival labels from {self.meta_path}...")
            with open(self.meta_path, "rb") as f:
                data = pickle.load(f)
            missing = _REQUIRED_CACHE_KEYS - set(data)
            if missing or data.get("label_contract_version") != LABEL_CONTRACT_VERSION:
                print(f"Cached survival labels incompatible (missing={sorted(missing)}). Rebuilding...")
            else:
                for k, v in data.items():
                    setattr(self, k, v)
                rebuild = False

        if rebuild:
            if self.verbose:
                print("Building operating-spell labels from source...")
            self.event_registry = self._get_event_registry()
            self._extract_spells()
            self._generate_survival_indices()
            save_dict = {
                "label_contract_version": LABEL_CONTRACT_VERSION,
                "source_fingerprint": self.source_fingerprint,
                "spell_table": self.spell_table,
                "split_indices": self.split_indices,
                "event_registry": self.event_registry,
                "spell_audit": self.spell_audit,
            }
            # Older caches (survival_meta_*.pkl) are intentionally left on disk
            # under their original names for auditability (§3.5).
            with atomic_path(self.meta_path) as tmp, open(tmp, "wb") as f:
                pickle.dump(save_dict, f)

        self._index_spells()

    # ------------------------------------------------------------------ #
    #  Spell extraction (§3.3)                                            #
    # ------------------------------------------------------------------ #

    def _extract_spells(self):
        """Build the physical operating-spell table and link terminating records.

        Columns (numpy arrays, one entry per spell, sorted by (line, start)):
            line, start, end (first outage hour or T), phys_reason
            (weather_event / non_weather_outage / end_of_data), term_record
            (registry index of the terminating outage if unique, else -1),
            term_status, weather_record (registry index of the weather target
            if unique, else -1), weather_status, ambiguous_causes,
            left_truncated.
        """
        aom = self.all_outages_matrix
        om = self.outage_matrix
        T, N = aom.shape

        lines, starts, ends = [], [], []
        for l in range(N):
            col = aom[:, l].astype(np.int8)
            padded = np.concatenate(([1], col, [1])).astype(np.int8)
            d = np.diff(padded)
            s = np.flatnonzero(d == -1)
            e = np.flatnonzero(d == 1)
            # every operating run has a start and an end (padding guarantees this)
            lines.append(np.full(len(s), l, dtype=np.int32))
            starts.append(s.astype(np.int64))
            ends.append(e.astype(np.int64))
        line = np.concatenate(lines)
        start = np.concatenate(starts)
        end = np.concatenate(ends)

        phys = np.full(len(line), TERM_END_OF_DATA, dtype=np.int8)
        inside = end < T
        weather_onset = np.zeros(len(line), dtype=bool)
        weather_onset[inside] = om[end[inside], line[inside]] == 1
        phys[inside & weather_onset] = TERM_WEATHER
        phys[inside & ~weather_onset] = TERM_NON_WEATHER

        reg = self.event_registry
        # One line episode per (line, onset hour): FeatureManager.outage_events merges all
        # mapped records starting in the same clock hour on a line (ID = earliest Outage
        # ID, all IDs kept). A spell end therefore links to at most one episode, whose
        # primary record is the terminating outage (and the weather target if weather).
        prim = reg[(reg["mapping_status"] == MAPPED) & reg["episode_primary"].astype(bool)]
        episode_at = {
            (int(li), int(t0)): (int(ridx), bool(w), int(nrec), str(ids), str(causes))
            for ridx, li, t0, w, nrec, ids, causes in zip(
                prim["registry_idx"], prim["line_idx"], prim["t_start"], prim["episode_weather"],
                prim["episode_n_records"], prim["episode_outage_ids"], prim["episode_causes"])
        }

        n = len(line)
        term_record = np.full(n, -1, dtype=np.int64)
        weather_record = np.full(n, -1, dtype=np.int64)
        term_status = np.full(n, ID_NOT_APPLICABLE, dtype=object)
        weather_status = np.full(n, ID_NOT_APPLICABLE, dtype=object)
        ambiguous_causes = np.zeros(n, dtype=bool)
        audit_rows = []
        wset = set(WEATHER_CAUSES)

        for i in np.flatnonzero(inside):
            ep = episode_at.get((int(line[i]), int(end[i])))
            if ep is None:
                term_status[i] = ID_NO_CANDIDATE
                if phys[i] == TERM_WEATHER:
                    weather_status[i] = ID_NO_CANDIDATE
                audit_rows.append({"spell_row": int(i), "line_idx": int(line[i]),
                                   "XRefCd": str(self.line_ids[int(line[i])]), "onset_t": int(end[i]),
                                   "phys_reason": TERMINATION_REASONS[int(phys[i])], "issue": ID_NO_CANDIDATE,
                                   "episode_outage_ids": "", "episode_causes": ""})
                continue
            ridx, is_w, nrec, ids, causes = ep
            term_status[i] = ID_UNIQUE
            term_record[i] = ridx
            if phys[i] == TERM_WEATHER:
                weather_status[i] = ID_UNIQUE if is_w else ID_NO_CANDIDATE
                weather_record[i] = ridx if is_w else -1
            cause_set = set(causes.split("|"))
            ambiguous_causes[i] = bool(cause_set & wset) and bool(cause_set - wset)
            if "|" in ids or ambiguous_causes[i]:
                # informational: merged by the agreed same-hour rule, not unresolved
                audit_rows.append({
                    "spell_row": int(i), "line_idx": int(line[i]), "XRefCd": str(self.line_ids[int(line[i])]),
                    "onset_t": int(end[i]), "phys_reason": TERMINATION_REASONS[int(phys[i])],
                    "issue": ("merged_same_hour_simultaneous_weather_and_non_weather" if ambiguous_causes[i]
                              else "merged_same_hour_records"),
                    "episode_outage_ids": ids, "episode_causes": causes,
                })

        self.spell_table = {
            "spell_id": np.arange(n, dtype=np.int64),
            "line": line, "start": start, "end": end, "phys_reason": phys,
            "term_record": term_record, "term_status": term_status,
            "weather_record": weather_record, "weather_status": weather_status,
            "ambiguous_causes": ambiguous_causes, "left_truncated": start == 0,
        }

        # Registry back-links: which spell (if any) each record terminates.
        reg = self.event_registry
        reg["operating_spell_id"] = -1
        reg["terminates_spell"] = False
        lookup = {(int(line[i]), int(end[i])): int(i) for i in np.flatnonzero(inside)}
        sp = [lookup.get((int(li), int(t0)), -1) if st == MAPPED else -1
              for li, t0, st in zip(reg["line_idx"], reg["t_start"].fillna(-1), reg["mapping_status"])]
        reg["operating_spell_id"] = np.asarray(sp, dtype=np.int64)
        reg["terminates_spell"] = reg["operating_spell_id"] >= 0
        # A mapped weather record whose onset happens while the line is already
        # out cannot terminate an operating spell (no event while not operating).
        unresolved_spells = set(np.flatnonzero(inside & (phys == TERM_WEATHER) & (weather_status != ID_UNIQUE)).tolist())
        reg["episode_note"] = np.where(
            (reg["mapping_status"] == MAPPED) & ~reg["terminates_spell"], "onset_while_line_not_operating",
            np.where(reg["operating_spell_id"].isin(list(unresolved_spells)), "ambiguous_mapping", None),
        )
        self.event_registry = reg
        self.spell_audit = pd.DataFrame(audit_rows)

    def _index_spells(self):
        """Search keys for vectorised (t, line) -> spell lookup."""
        st = self.spell_table
        self._spell_key = st["line"].astype(np.int64) * (self.total_t + 1) + st["start"].astype(np.int64)

    # ------------------------------------------------------------------ #
    #  Vectorised label construction                                      #
    # ------------------------------------------------------------------ #

    def split_of_time(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        return np.where(t < self.train_end_idx, 0, np.where(t < self.val_end_idx, 1, 2)).astype(np.int64)

    def labels_at(self, t_predict: np.ndarray) -> dict[str, np.ndarray]:
        """Survival labels for every line at each prediction time (flattened t-major).

        Returns arrays of length len(t_predict) * N_lines with:
            prediction_time, line_idx, split_code, eligible, exclusion_code,
            tte (capped U), event (capped), termination_code (capped),
            raw_follow_up, raw_termination_code, operating_spell_id,
            next_outage_record, target_weather_record, id_status_code
        """
        t = np.asarray(t_predict, dtype=np.int64).reshape(-1)
        N = self.n_lines
        T = self.total_t
        H = float(self.max_horizon)
        tt = np.repeat(t, N)
        ll = np.tile(np.arange(N, dtype=np.int64), len(t))

        eligible = self.all_outages_matrix[tt, ll] == 0
        split_code = self.split_of_time(tt)
        split_end = np.where(split_code == 0, self.train_end_idx, np.where(split_code == 1, self.val_end_idx, T))

        st = self.spell_table
        s = np.searchsorted(self._spell_key, ll * (T + 1) + tt, side="right") - 1
        s = np.clip(s, 0, len(self._spell_key) - 1)
        s_end = st["end"][s]
        phys = st["phys_reason"][s].astype(np.int64)

        # The split observes hours [.., split_end). An onset AT split_end belongs to
        # the next split, so the segment is censored at the boundary (half-open).
        within = s_end < split_end
        raw_term = np.where(within, phys, np.where(split_end >= T, TERM_END_OF_DATA, TERM_SPLIT))
        obs_end = np.minimum(s_end, split_end)
        raw_u = (obs_end - tt).astype(np.float64)

        capped = raw_u >= H
        tte = np.where(capped, H, raw_u)
        event = (raw_term == TERM_WEATHER) & ~capped
        term = np.where(capped, TERM_MAX_H, raw_term)

        # The terminating outage of the observable segment (any cause), if any.
        ends_in_outage = within & ((phys == TERM_WEATHER) | (phys == TERM_NON_WEATHER))
        next_rec = np.where(ends_in_outage, st["term_record"][s], -1)
        target = np.where(event, st["weather_record"][s], -1)

        # Ineligible rows: no follow-up, no IDs, explicit exclusion reason.
        tte = np.where(eligible, tte, 0.0)
        raw_u = np.where(eligible, raw_u, 0.0)
        event = event & eligible
        term = np.where(eligible, term, TERM_INELIGIBLE)
        raw_term = np.where(eligible, raw_term, TERM_INELIGIBLE)
        spell_id = np.where(eligible, st["spell_id"][s], -1)
        next_rec = np.where(eligible, next_rec, -1)
        target = np.where(eligible, target, -1)

        return {
            "prediction_time": tt, "line_idx": ll, "split_code": split_code,
            "eligible": eligible, "exclusion_code": np.where(eligible, EXCL_NONE, EXCL_ALREADY_OUTAGED),
            "tte": tte.astype(np.float32), "event": event,
            "termination_code": term.astype(np.int64),
            "raw_follow_up": raw_u.astype(np.float32), "raw_termination_code": raw_term.astype(np.int64),
            "operating_spell_id": spell_id.astype(np.int64),
            "next_outage_record": next_rec.astype(np.int64), "target_weather_record": target.astype(np.int64),
        }

    def label_table(self, split: str, indices=None) -> pd.DataFrame:
        """Observation labels for all lines at a split's scheduled prediction times.

        No weather I/O. Used for the shared observation table (§4), the IPCW
        censoring reference (§6.3), event denominators (§7) and the audit (§3.5).
        """
        idx = self.split_indices[split] if indices is None else indices
        t_predict = np.asarray(idx, dtype=np.int64) + int(self.input_win)
        lab = self.labels_at(t_predict)
        df = pd.DataFrame(lab)
        df["split"] = split
        return df

    # ------------------------------------------------------------------ #
    #  Sampling Indices                                                   #
    # ------------------------------------------------------------------ #

    def _generate_survival_indices(self):
        """Scheduled prediction-start indices per split.

        Unchanged from the previous implementation so the sampled timestamps of
        the saved reference run are reproduced exactly (§2.1). Note: the last
        admissible start (t_start == split_end - input_win - H) is excluded by
        this conservative stop; recorded in the contract rather than changed.
        """
        total_t = self.outage_matrix.shape[0]
        window_len = self.input_win + self.max_horizon

        train_stop = self.train_end_idx - window_len
        train_stop = max(0, min(train_stop, total_t - window_len))

        val_start = self.train_end_idx
        val_stop = self.val_end_idx - window_len
        val_stop = max(val_start, min(val_stop, total_t - window_len))

        test_start = self.val_end_idx
        test_stop = total_t - window_len

        self.split_indices = {
            "train": list(range(0, train_stop, self.train_stride)),
            "val": list(range(val_start, val_stop, self.val_stride)),
            "test": list(range(test_start, test_stop, self.test_stride)),
        }

        if self.verbose:
            for split, indices in self.split_indices.items():
                print(f"  {split}: {len(indices)} samples (stride={getattr(self, f'{split}_stride', '?')})")

    # ------------------------------------------------------------------ #
    #  Weather-to-weather adjacency (§9.5 pilot)                          #
    # ------------------------------------------------------------------ #

    def _build_weather_grid_edges(self, mode: str = "grid8"):
        """Explicit grid adjacency over the regular weather grid (decided 2026-10-08).

        Cells are mapped to integer grid indices using the grid step (0.1 degree).
        Each cell is linked, in both directions, to the cells that EXIST among its
        4 (grid4: up/down/left/right) or 8 (grid8: + diagonals) grid neighbours, so
        a full corner has 3 neighbours, a full edge 5 and an interior cell 8; fewer
        where neighbouring cells are absent. No self loops. Edge attributes are the
        (dlat, dlon) offsets in grid steps, i.e. values in {-1, 0, 1}. Built from
        static cell geometry only (no outage information).
        """
        cells = np.asarray(self.unique_weather_cells, dtype=np.float64)  # (lon, lat)
        if len(cells) < 2:
            self.weather_weather_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.weather_weather_edge_attr = torch.zeros((0, 2), dtype=torch.float)
            return

        def step(v):
            d = np.diff(np.unique(np.round(v, 6)))
            return float(d[d > 1e-9].min()) if (d > 1e-9).any() else 1.0

        lon_step, lat_step = step(cells[:, 0]), step(cells[:, 1])
        gx = np.round((cells[:, 0] - cells[:, 0].min()) / lon_step).astype(np.int64)
        gy = np.round((cells[:, 1] - cells[:, 1].min()) / lat_step).astype(np.int64)
        index = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(gx, gy))}
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if mode == "grid8":
            offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        src, dst, attr = [], [], []
        for i, (x, y) in enumerate(zip(gx, gy)):
            for dx, dy in offsets:
                j = index.get((int(x + dx), int(y + dy)))
                if j is not None:
                    src.append(i)
                    dst.append(j)
                    attr.append((float(dy), float(dx)))  # (dlat, dlon) in grid steps
        self.weather_weather_edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
        self.weather_weather_edge_attr = torch.tensor(attr, dtype=torch.float).reshape(-1, 2)

    @property
    def graph_version(self) -> str:
        return f"wl{tuple(self.weather_edges_index.shape)}_ll{tuple(self.line_edges_index.shape)}_ww-{self.weather_adjacency}"

    # ------------------------------------------------------------------ #
    #  Sample Construction                                                #
    # ------------------------------------------------------------------ #

    def construct_sample(self, t_start):
        """HeteroData snapshot with survival targets and observation metadata.

        Inputs are identical to OutageDataManager. Targets (all per line node):
            eligible (N,1) bool            operating at prediction time (§3.1)
            tte, event (N,1)               horizon-capped follow-up / event
            discrete_labels, discrete_mask (N,K)   likelihood targets (zero if ineligible)
            cum_label, known_outcome (N,K) cumulative horizon outcomes (§3.2)
            termination_code, raw_follow_up, raw_termination_code (N,)
            operating_spell_id, next_outage_record, target_weather_record (N,)
            prediction_time, line_idx, split_code (N,)  join keys for exports
        Attribute names avoid 'index'/'batch' so PyG does not offset them on batching.
        Outcome/future fields are evaluation metadata and never model inputs.
        """
        t_predict = t_start + self.input_win
        data = HeteroData()
        n_lines = len(self.line_ids)
        data["line"].num_nodes = n_lines

        data["line"].x_static = self.line_features_static
        data["line"].xref_idx = self.line_xref_idx

        status_hist = self.outage_matrix[t_start:t_predict, :]
        time_since_hist = self.time_since_matrix[t_start:t_predict, :]
        combined_dynamic = np.stack([status_hist, time_since_hist], axis=-1)
        data["line"].x_dynamic = torch.from_numpy(combined_dynamic).float().permute(1, 0, 2)

        weather_feats = self.base.get_weather_tensor_for_window(t_start, self.input_win)
        data["weather"].x = weather_feats.permute(1, 0, 2)
        data["weather"].num_nodes = data["weather"].x.size(0)

        data["weather", "affects", "line"].edge_index = self.weather_edges_index
        data["weather", "affects", "line"].edge_attr = self.weather_edges_attr
        data["line", "connected_to", "line"].edge_index = self.line_edges_index
        data["line", "connected_to", "line"].edge_attr = self.line_edges_attr
        if self.weather_weather_edge_index is not None:
            data["weather", "near", "weather"].edge_index = self.weather_weather_edge_index
            data["weather", "near", "weather"].edge_attr = self.weather_weather_edge_attr

        lab = self.labels_at(np.array([t_predict]))
        eligible = lab["eligible"]
        labels, mask = discrete_targets(lab["tte"], lab["event"], eligible, self.interval_bounds)
        outcomes = horizon_outcomes(lab["tte"], lab["event"], eligible, self.interval_bounds)

        line = data["line"]
        line.eligible = torch.from_numpy(eligible).bool().unsqueeze(-1)
        line.tte = torch.from_numpy(lab["tte"]).float().unsqueeze(-1)
        line.event = torch.from_numpy(lab["event"].astype(np.float32)).unsqueeze(-1)
        line.discrete_labels = torch.from_numpy(labels)
        line.discrete_mask = torch.from_numpy(mask)
        line.cum_label = torch.from_numpy(outcomes["positive"].astype(np.float32))
        line.known_outcome = torch.from_numpy(outcomes["known_outcome"])
        line.termination_code = torch.from_numpy(lab["termination_code"])
        line.raw_follow_up = torch.from_numpy(lab["raw_follow_up"])
        line.raw_termination_code = torch.from_numpy(lab["raw_termination_code"])
        line.operating_spell_id = torch.from_numpy(lab["operating_spell_id"])
        line.next_outage_record = torch.from_numpy(lab["next_outage_record"])
        line.target_weather_record = torch.from_numpy(lab["target_weather_record"])
        line.prediction_time = torch.from_numpy(lab["prediction_time"])
        line.line_idx = torch.from_numpy(lab["line_idx"])
        line.split_code = torch.from_numpy(lab["split_code"])

        # Legacy binary label (weather outage hours in the next output_win).
        t_end = t_predict + self.output_win
        if t_end <= self.outage_matrix.shape[0]:
            y_binary = np.max(self.outage_matrix[t_predict:t_end, :], axis=0)
        else:
            y_binary = np.zeros(n_lines, dtype=np.float32)
        line.y_binary = torch.from_numpy(np.asarray(y_binary, dtype=np.float32)).unsqueeze(-1)
        return data

    # ------------------------------------------------------------------ #
    #  DataLoader Creation                                                #
    # ------------------------------------------------------------------ #

    def get_dataset(self, mode="train"):
        if mode not in self.split_indices:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.split_indices.keys())}")
        return SurvivalDataset(self, mode)

    def create_dataloaders(self, batch_size=32, num_workers=4, pin_memory=True, prefetch_factor=2,
                           generator: Optional[torch.Generator] = None):
        """Train loader is shuffled (seeded via `generator`); val/test keep schedule order.

        Exports never rely on loader order: every row carries prediction_time and
        line_idx (§4.2).
        """
        loader_args = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
        if num_workers > 0 and prefetch_factor is not None:
            loader_args["prefetch_factor"] = prefetch_factor
        train_loader = DataLoader(self.get_dataset("train"), shuffle=True, generator=generator, **loader_args)
        val_loader = DataLoader(self.get_dataset("val"), shuffle=False, **loader_args)
        test_loader = DataLoader(self.get_dataset("test"), shuffle=False, **loader_args)
        return train_loader, val_loader, test_loader

    def compute_interval_weights(self, train_loader=None, cap=50.0):
        """Per-interval event weights alpha_k = sqrt(at_risk_k / events_k), capped.

        Used by the DeepHit/DeepSurv baselines only. The survival GNN trainer
        deliberately does not weight events (§8.1). Computed from labels
        directly (no weather I/O); `train_loader` kept for API compatibility.
        """
        weights_path = os.path.join(self.processed_dir, f"interval_weights_{self.config_hash}.pt")
        if os.path.exists(weights_path):
            if self.verbose:
                print(f"Loading cached interval weights from {weights_path}...")
            return torch.load(weights_path)
        lab = self.label_table("train")
        labels, mask = discrete_targets(lab["tte"], lab["event"], lab["eligible"], self.interval_bounds)
        at_risk = torch.from_numpy(mask.sum(0))
        events = torch.from_numpy((labels * mask).sum(0))
        alpha = torch.sqrt(at_risk / events.clamp(min=1)).clamp(max=cap)
        torch.save(alpha, weights_path)
        return alpha

    # ------------------------------------------------------------------ #
    #  Exports                                                            #
    # ------------------------------------------------------------------ #

    def spell_dataframe(self) -> pd.DataFrame:
        st = self.spell_table
        df = pd.DataFrame({k: v for k, v in st.items()})
        df["XRefCd"] = [self.line_ids[int(i)] for i in df["line"]]
        df["phys_reason"] = [TERMINATION_REASONS[int(r)] for r in df["phys_reason"]]
        df["duration"] = df["end"] - df["start"]
        return df

    def export_tabular(self, split="train", include_weather_summary=False):
        """One row per spell segment inside a split (legacy helper for classical models)."""
        lo, hi = self.split_bounds[split]
        sp = self.spell_dataframe()
        sp = sp[(sp["end"] > lo) & (sp["start"] < hi)].copy()
        sp["t_start"] = sp["start"].clip(lower=lo)
        sp["t_end"] = sp["end"].clip(upper=hi)
        cut = sp["end"] > hi
        sp["event"] = ((sp["phys_reason"] == "weather_event") & ~cut).astype(int)
        sp["censor_reason"] = np.where(
            sp["event"] == 1, None,
            np.where(cut, np.where(hi >= self.total_t, "end_of_data", "split_boundary"), sp["phys_reason"]),
        )
        sp["duration"] = sp["t_end"] - sp["t_start"]
        static = self.line_features_static.numpy()
        for i, name in enumerate(["voltage", "length", "type_L", "type_T", "type_TT"][: static.shape[1]]):
            sp[name] = static[sp["line"].to_numpy(), i]
        return sp.rename(columns={"line": "line_idx"}).reset_index(drop=True)

    def data_audit(self, splits=SPLITS) -> dict[str, Any]:
        """Per-split counts required by §3.5 (no model predictions involved)."""
        reg = self.event_registry
        out: dict[str, Any] = {
            "label_contract_version": LABEL_CONTRACT_VERSION,
            "source_fingerprint": self.source_fingerprint,
            "max_horizon": self.max_horizon,
            "discrete_intervals": self.discrete_intervals,
            "registry_mapping_status": reg["mapping_status"].value_counts().to_dict(),
            "registry_data_quality_flags": reg["data_quality_flag"].dropna().value_counts().to_dict(),
            "registry_records_merged_into_episodes": int(reg["duplicate_of"].notna().sum()),
            "episodes_still_out_to_end_of_data": int(((reg["episode_restore_status"] == "still_out_extended_to_end_of_data")
                                                      & reg["episode_primary"].astype(bool)).sum()),
            "weather_causes": list(WEATHER_CAUSES),
            "spells_total": int(len(self.spell_table["spell_id"])),
            "spell_id_issues": (self.spell_audit["issue"].value_counts().to_dict() if len(self.spell_audit) else {}),
            "splits": {},
        }
        for split in splits:
            lab = self.label_table(split)
            el = lab["eligible"].to_numpy()
            term = lab["termination_code"].to_numpy()
            ev = lab["event"].to_numpy()
            reasons = {}
            for code, name in enumerate(TERMINATION_REASONS):
                reasons[name] = int(((term == code) & el).sum())
            tgt = lab.loc[ev, "target_weather_record"].to_numpy()
            lep = reg.set_index("registry_idx").reindex(tgt[tgt >= 0])
            # event mix by weather cause (episode's weather cause(s), e.g. 'Lightning')
            wset = set(WEATHER_CAUSES)
            ep_cause = lep["episode_causes"].map(
                lambda c: "+".join(x for x in str(c).split("|") if x in wset) or "unresolved") if len(lep) else pd.Series(dtype=object)
            out["splits"][split] = {
                "prediction_times": int(len(self.split_indices[split])),
                "stride": int(getattr(self, f"{split}_stride")),
                "node_rows": int(len(lab)),
                "eligible_rows": int(el.sum()),
                "ineligible_rows": int((~el).sum()),
                "event_rows": int(ev.sum()),
                "termination_reasons_eligible": reasons,
                "operating_spells": int(np.unique(lab.loc[el, "operating_spell_id"]).size),
                "target_line_episodes": int(lep["line_episode_id"].nunique()) if len(lep) else 0,
                "target_incident_episodes": int(lep["incident_episode_id"].nunique()) if len(lep) else 0,
                "event_rows_with_unresolved_target_id": int((tgt < 0).sum()),
                "event_rows_by_cause": ep_cause.value_counts().to_dict() if len(lep) else {},
                "episodes_by_cause": (lep.assign(c=ep_cause.to_numpy()).drop_duplicates("line_episode_id")["c"]
                                      .value_counts().to_dict() if len(lep) else {}),
            }
        return out

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #

    def _print_summary(self):
        print("\n" + "=" * 65)
        print("  SURVIVAL OUTAGE DATA MANAGER — Summary (operating spells)")
        print("=" * 65)
        print(f"  Label contract: {LABEL_CONTRACT_VERSION}  (cache {os.path.basename(self.meta_path)})")
        print(f"  Lines: {len(self.line_ids)}")
        print(f"  Max horizon: {self.max_horizon}h, intervals: {self.discrete_intervals} "
              f"(width={self.interval_width:.1f}h), input window: {self.input_win}h")
        sp = self.spell_dataframe()
        for split in SPLITS:
            lo, hi = self.split_bounds[split]
            seg = sp[(sp["end"] > lo) & (sp["start"] < hi)]
            ended = seg[seg["end"] <= hi]
            print(f"\n  {split.upper()}: {len(self.split_indices.get(split, []))} prediction times; "
                  f"{len(seg)} spell segments; terminations inside split: "
                  + ", ".join(f"{r}={c}" for r, c in ended["phys_reason"].value_counts().items()))
        if len(self.spell_audit):
            print(f"\n  ID-linkage issues (audit table): {self.spell_audit['issue'].value_counts().to_dict()}")
        print("=" * 65)


# ============================================================================ #
#  LEGACY LOSS FUNCTIONS (kept for the DeepHit / DeepSurv baseline scripts)    #
#  The survival GNN trainer uses SurvivalModels/losses.py (§8).                #
# ============================================================================ #

class DiscreteSurvivalLoss(torch.nn.Module):
    """Masked discrete-time survival NLL on probabilities (legacy; see losses.py)."""

    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha

    def forward(self, hazard_pred, labels, mask):
        eps = 1e-7
        hazard_pred = hazard_pred.clamp(eps, 1.0 - eps)
        log_h = torch.log(hazard_pred)
        log_1_minus_h = torch.log(1.0 - hazard_pred)
        if self.alpha is not None:
            ll = (self.alpha.to(labels.device) * labels * log_h) + (1.0 - labels) * log_1_minus_h
        else:
            ll = labels * log_h + (1.0 - labels) * log_1_minus_h
        ll = ll * mask
        return -ll.sum() / mask.sum().clamp(min=1.0)


class RankingRegularization(torch.nn.Module):
    """Legacy pairwise ranking term used by the DeepHit/DeepSurv scripts.

    Known limitations (fixed in losses.ComparablePairRankingLoss, used by the GNN):
    first-interval mask as eligibility, (batch, N, K) shape assumption,
    `len(event_idx) < 2` shortcut, unseeded sampling, no within-spell exclusion.
    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, risk_scores, tte, event, mask):
        risk = risk_scores.view(-1)
        t = tte.view(-1)
        e = event.view(-1)
        m = mask[:, :, 0].reshape(-1).bool()
        risk, t, e = risk[m], t[m], e[m]
        event_idx = torch.where(e == 1)[0]
        if len(event_idx) < 2:
            return risk.sum() * 0.0
        n_pairs = min(len(event_idx) * 10, 5000)
        idx_i = event_idx[torch.randint(len(event_idx), (n_pairs,), device=risk.device)]
        idx_j = torch.randint(len(risk), (n_pairs,), device=risk.device)
        valid = t[idx_j] > t[idx_i]
        if valid.sum() == 0:
            return risk.sum() * 0.0
        idx_i, idx_j = idx_i[valid], idx_j[valid]
        diff = risk[idx_i] - risk[idx_j]
        return torch.mean(torch.nn.functional.softplus(-diff / self.sigma))


class OrdinalSurvivalLoss(torch.nn.Module):
    """Legacy DNLL + ordinal_weight * event-only RPS (single combined scalar).

    The GNN trainer now uses losses.SurvivalObjective, which logs DNLL and RPS
    separately. Kept unchanged so the DeepHit/DeepSurv baselines reproduce.
    """

    def __init__(self, alpha=None, ordinal_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.ordinal_weight = ordinal_weight

    def forward(self, hazard_pred, labels, mask):
        eps = 1e-7
        hazard_pred = hazard_pred.clamp(eps, 1.0 - eps)
        log_h = torch.log(hazard_pred)
        log_1_minus_h = torch.log(1.0 - hazard_pred)
        if self.alpha is not None:
            ll = self.alpha.to(labels.device) * labels * log_h + (1.0 - labels) * log_1_minus_h
        else:
            ll = labels * log_h + (1.0 - labels) * log_1_minus_h
        ll = ll * mask
        nll = -ll.sum() / mask.sum().clamp(min=1.0)
        if self.ordinal_weight > 0:
            F_pred = 1.0 - torch.exp(torch.cumsum(log_1_minus_h, dim=-1))
            F_true = torch.cumsum(labels, dim=-1)
            has_event = (labels.sum(dim=-1, keepdim=True) > 0.5).float()
            if has_event.sum() > 0:
                rps = torch.mean((F_pred - F_true) ** 2, dim=-1)
                rps_loss = (rps * has_event.squeeze(-1)).sum() / has_event.sum().clamp(min=1.0)
            else:
                rps_loss = torch.tensor(0.0, device=hazard_pred.device)
        else:
            rps_loss = torch.tensor(0.0, device=hazard_pred.device)
        return nll + self.ordinal_weight * rps_loss
