"""
Evaluation / label contract for the operating-spell weather-outage study.

IMPLEMENTATION_STEPS.md work packages served here:
    §2.2  Immutable, versioned evaluation contract (saved with every run)
    §3.2  Half-open interval convention and the per-horizon outcome definitions
    §5.1  Fixed interval grid and cumulative-risk helpers
    §6.4  Float64 left-limit evaluation times (b_k^-)

Everything that decides *who is in the population* or *what a label means* lives
here, in one place, so the data manager, the trainer, the evaluator and the
analysis scripts cannot drift apart. A change to any value in
`EvaluationContract` must bump `contract_version`; model-only changes reuse it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
#  Versions                                                                    #
# --------------------------------------------------------------------------- #

# Bump when eligibility, termination precedence, ID linkage or the interval
# convention changes. Part of every survival cache key (§3.5).
# v2: expanded weather causes, consolidated CSV, same-hour merge
# v3: BPA Pacific times converted to UTC; real 2021 file; 2021-2023 rebuilt from BPA's xlsx (R5)
LABEL_CONTRACT_VERSION = "operating_spell_labels_v3"

# Bump when a numerical metric definition changes (§10.4 metric_version).
# v2: IPCW reported with training and evaluation-set references; v3: val_dnll selector, degenerate-h1 guard
METRIC_VERSION = "operating_spell_metrics_v3"

# Bump when the loss arithmetic changes (§8.1 logit-based DNLL).
LOSS_VERSION = "dnll_logit_v1"

# Weather causes used for the event definition: ONE list, defined (with the
# rationale and possible expansions) in FeatureManager/outage_events.py and used
# by the matcher, the outage matrices and the labels alike.
from FeatureManager.outage_events import DEFAULT_WEATHER_CAUSES as WEATHER_CAUSES  # noqa: E402

# --------------------------------------------------------------------------- #
#  Normalised termination / exclusion codes (§3.3)                             #
# --------------------------------------------------------------------------- #

# Stored as small ints inside tensors; names are what exports/reports show.
TERMINATION_REASONS: tuple[str, ...] = (
    "weather_event",       # 0  observed weather outage onset (the event)
    "non_weather_outage",  # 1  competing termination -> censoring
    "split_boundary",      # 2  administrative: end of train/val period
    "end_of_data",         # 3  administrative: end of the whole dataset
    "max_horizon",         # 4  follow-up reached H without termination
)
TERM_WEATHER, TERM_NON_WEATHER, TERM_SPLIT, TERM_END_OF_DATA, TERM_MAX_H = range(5)
TERM_INELIGIBLE = -1  # row is not in the survival population at all

ADMINISTRATIVE_TERMINATIONS = (TERM_SPLIT, TERM_END_OF_DATA)

EXCLUSION_REASONS: tuple[str, ...] = (
    "",                 # 0  eligible
    "already_outaged",  # 1  line is in (any) outage at prediction time
)
EXCL_NONE, EXCL_ALREADY_OUTAGED = 0, 1


def termination_name(code: int) -> Optional[str]:
    code = int(code)
    if code < 0:
        return None
    return TERMINATION_REASONS[code]


def censor_reason_name(term_code: int, event: bool) -> Optional[str]:
    """§3.3: `censor_reason` is null for an event (events are not censoring)."""
    if event or int(term_code) < 0:
        return None
    return TERMINATION_REASONS[int(term_code)]


# --------------------------------------------------------------------------- #
#  Grid helpers (§5.1, §6.4)                                                   #
# --------------------------------------------------------------------------- #

def interval_bounds(max_horizon: float, n_intervals: int) -> np.ndarray:
    """[b_0=0, b_1, ..., b_K=H] as float64. Built once from the configured H."""
    if max_horizon is None or float(max_horizon) <= 0:
        raise ValueError(f"max_horizon must be a positive number, got {max_horizon!r}")
    if int(n_intervals) < 1:
        raise ValueError(f"n_intervals must be >= 1, got {n_intervals!r}")
    return np.linspace(0.0, float(max_horizon), int(n_intervals) + 1, dtype=np.float64)


def evaluation_times(bounds: np.ndarray) -> np.ndarray:
    """Left limits b_k^- (k=1..K) for library calls on half-open horizons (§6.4)."""
    return np.nextafter(np.asarray(bounds[1:], dtype=np.float64), -np.inf)


def survival_from_hazard(hazard: np.ndarray) -> np.ndarray:
    """S_i(k) = prod_{j<=k} (1 - h_ij). Shape (N, K)."""
    hazard = np.asarray(hazard, dtype=np.float64)
    return np.cumprod(1.0 - hazard, axis=1)


def cumulative_risk(hazard: np.ndarray) -> np.ndarray:
    """R_i(k) = 1 - S_i(k): cumulative weather risk through horizon k (§5.1)."""
    return 1.0 - survival_from_hazard(hazard)


def full_curve_risk(hazard: np.ndarray) -> np.ndarray:
    """Scalar risk for concordance: -sum_k S_i(k) (§6.6).

    A discrete survival-curve summary (proportional to the restricted mean
    event-free time for equal-width bins), not an exact expected survival time.
    """
    return -survival_from_hazard(hazard).sum(axis=1)


def check_hazards(hazard: np.ndarray, n_intervals: int) -> None:
    """§5.1 assertions: finite, inside [0,1], width K, nondecreasing R."""
    hazard = np.asarray(hazard)
    if hazard.ndim != 2 or hazard.shape[1] != int(n_intervals):
        raise ValueError(f"Hazard array must be (N, {n_intervals}); got {hazard.shape}")
    if hazard.size == 0:
        return
    if not np.all(np.isfinite(hazard)):
        raise ValueError("Hazard predictions contain NaN/inf values.")
    if hazard.min() < 0.0 or hazard.max() > 1.0:
        raise ValueError(f"Hazards must be in [0,1]; got range [{hazard.min()}, {hazard.max()}]")
    risk = cumulative_risk(hazard)
    if np.any(np.diff(risk, axis=1) < -1e-12):
        raise ValueError("Cumulative risks are not nondecreasing across horizons.")


# --------------------------------------------------------------------------- #
#  Per-horizon outcome definitions (§3.2)                                      #
# --------------------------------------------------------------------------- #

def horizon_outcomes(
    follow_up: np.ndarray,
    event: np.ndarray,
    eligible: np.ndarray,
    bounds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Cumulative outcome masks, all (N, K) bool.

        positive_k        = eligible AND event AND (U < b_k)
        known_negative_k  = eligible AND (U >= b_k)
        known_outcome_k   = positive_k OR known_negative_k
        early_censored_k  = eligible AND (NOT event) AND (U < b_k)

    `follow_up` is the horizon-capped U (tte) and `event` the capped event flag.
    An event exactly at an internal bound b_k is a known negative for horizon k
    and a positive for k+1 (half-open intervals). Censoring exactly at b_k is a
    known event-free outcome through b_k.
    """
    u = np.asarray(follow_up, dtype=np.float64).reshape(-1, 1)
    e = np.asarray(event).astype(bool).reshape(-1, 1)
    el = np.asarray(eligible).astype(bool).reshape(-1, 1)
    b = np.asarray(bounds, dtype=np.float64)[1:].reshape(1, -1)
    before = u < b
    positive = el & e & before
    known_negative = el & ~before
    return {
        "positive": positive,
        "known_negative": known_negative,
        "known_outcome": positive | known_negative,
        "early_censored": el & ~e & before,
    }


def discrete_targets(
    follow_up: np.ndarray, event: np.ndarray, eligible: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-hazard labels and likelihood mask, (N, K) float32 each.

    Same semantics as the original `_compute_discrete_hazard`, vectorised and
    explicitly zeroed for ineligible rows (§3.1):
      event in [b_{k-1}, b_k):  label 1 at k, mask 1 for all j <= k
      censored at U:            mask 1 for every interval fully survived (U >= b_k)
    """
    u = np.asarray(follow_up, dtype=np.float64).reshape(-1, 1)
    e = np.asarray(event).astype(bool).reshape(-1, 1)
    el = np.asarray(eligible).astype(bool).reshape(-1, 1)
    lo = np.asarray(bounds, dtype=np.float64)[:-1].reshape(1, -1)
    hi = np.asarray(bounds, dtype=np.float64)[1:].reshape(1, -1)
    at_risk_event = u >= lo
    in_interval = (u >= lo) & (u < hi)
    labels = el & e & in_interval
    mask = el & np.where(e, at_risk_event, u >= hi)
    return labels.astype(np.float32), mask.astype(np.float32)


# --------------------------------------------------------------------------- #
#  The contract object (§2.2)                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvaluationContract:
    """Everything needed to reconstruct labels and metrics without current defaults."""

    contract_version: str = "operating_spell_eval_v3"
    label_contract_version: str = LABEL_CONTRACT_VERSION
    metric_version: str = METRIC_VERSION

    # -- grid -------------------------------------------------------------
    max_horizon: float = 1008.0
    discrete_intervals: int = 6
    time_unit: str = "hour"
    interval_convention: str = "half_open [b_(k-1), b_k); event at internal bound -> next interval; event at H -> censored at H"
    hourly_rounding: str = "outage start floored to the hour; recovery ceiled to the hour (FeatureManager)"

    # -- population -------------------------------------------------------
    eligibility_rule: str = "all_outages_matrix[t_predict, line] == 0 (operating at prediction time)"
    episode_rule: str = ("records on one line starting in the same clock hour form one episode: ID = earliest Outage ID "
                         "(all kept); restore = latest known restore ('still out' only if none known); weather if any")
    cause_rule: str = "per outage: weather cause if any row has one; else first non-Unknown; else source wording/Unknown"
    observability_rule: str = "source has no unknown-status code; all scheduled rows are treated as observed"
    termination_precedence: str = (
        "earliest of weather onset, non-weather onset, split boundary, end of data; "
        "simultaneous weather+non-weather onset in one hour is one episode labelled weather (audited)"
    )
    comparable_pair_rule: str = "event_i and U_i < U_j, both eligible, different operating spells"
    weather_causes: tuple[str, ...] = WEATHER_CAUSES

    # -- sampling ---------------------------------------------------------
    split_strides: dict[str, int] = field(default_factory=lambda: {"train": 24, "val": 24, "test": 24})

    # -- thresholds / budgets ---------------------------------------------
    fpr_targets: tuple[float, ...] = (0.001, 0.01)
    top_fraction: float = 0.01
    daily_budgets: tuple[int, ...] = (5, 10)
    threshold_rule: str = "alert iff score >= threshold; tied scores always share a decision"
    threshold_tie_break: str = "max recall, then fewer FP, then stricter threshold"
    top_b_tie_break: str = "descending score, then ascending stable line index"
    risk_tie_tolerance: float = 1e-12
    ece_bins: int = 20

    # -- daily decision policy --------------------------------------------
    source_timezone: Optional[str] = "America/Los_Angeles"
    source_timezone_status: str = ("BPA outage times are Pacific wall clock (PST/PDT; R5.2, 2026-10-19), converted to "
                                   "tz-naive UTC in the matcher (FeatureManager.outage_events.bpa_pacific_to_utc); "
                                   "the weather index and every label/prediction time are UTC")
    daily_issue_hour: int = 13
    daily_issue_hour_status: str = ("user selected 2026-10-08: 13:00 UTC = 05:00 PST / 06:00 PDT; evaluated on test "
                                    "(stride 1); validation (stride 24, 04:00 UTC) has no 13:00 snapshot")
    min_lead_hours: float = 1.0
    min_lead_status: str = ("user selected 2026-10-08: next-hour lead (any alert before the onset hour counts); "
                            "longer operational lead requirements are a documented future expansion")

    # -- IPCW / IBS -------------------------------------------------------
    ipcw_reference: str = ("reported twice: *_ipcw_trainref (eligible training rows at the scheduled training times) "
                           "and *_ipcw_evalref (eligible rows of the evaluated split itself; censoring differs by split). "
                           "Legacy val_c_index maps to c_index_ipcw_evalref")
    censoring_estimator: str = "reverse Kaplan-Meier, events-before-censoring at tied times"
    ipcw_event_weight_convention: str = "left_limit"  # G(U_i^-) (Graf et al. 1999); 'library' = G(U_i)
    ibs_convention: str = "trapezoid over (0,0),(b_1,BS_1)..(b_K,BS_K), divided by H"
    missing_metric_behaviour: str = "value=null plus status reason; never substituted"

    # -- checkpoint selection (§10.2) ---------------------------------------
    selectors: tuple[str, ...] = ("reference_selector", "recall_fpr_001_h1", "partial_auc_001_h1", "dnll")
    selector_tie_break: str = "higher target metric, then lower validation DNLL, then earlier epoch"
    # An epoch whose validation R_1 has a standard deviation below this over eligible rows is a
    # near-constant (input-independent) first-horizon output: the h1 selectors may not choose it,
    # whole-curve selectors may (the epoch is flagged in history and report) (v3).
    degenerate_score_std: float = 1e-4

    # ------------------------------------------------------------------ #
    @property
    def bounds(self) -> np.ndarray:
        return interval_bounds(self.max_horizon, self.discrete_intervals)

    @property
    def eval_times(self) -> np.ndarray:
        return evaluation_times(self.bounds)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["interval_bounds"] = self.bounds.tolist()
        d["library_evaluation_times"] = [float(x) for x in self.eval_times]
        return d

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        from FeatureManager.atomic_io import atomic_write_text
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint()
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationContract":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in allowed}
        for key in ("fpr_targets", "daily_budgets", "weather_causes", "selectors"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationContract":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def assert_compatible(self, *, max_horizon: float, discrete_intervals: int, source: str) -> None:
        """§2.1: model/data/checkpoint/evaluator horizon and K must agree exactly."""
        if not math.isclose(float(max_horizon), float(self.max_horizon)) or int(discrete_intervals) != int(
            self.discrete_intervals
        ):
            raise ValueError(
                f"Horizon mismatch between the evaluation contract (H={self.max_horizon}, "
                f"K={self.discrete_intervals}) and {source} (H={max_horizon}, K={discrete_intervals}). "
                "Labels, model outputs and metrics would refer to different intervals. Rebuild the data "
                "manager with the model's horizon, or evaluate under a contract matching the checkpoint."
            )


def assert_same_grid(*, sources: Sequence[tuple[str, Optional[float], Optional[int]]]) -> tuple[float, int]:
    """Assert equality of (H, K) across named sources; return the common pair."""
    known = [(n, h, k) for n, h, k in sources if h is not None and k is not None]
    if not known:
        raise ValueError("No source supplied a horizon/interval count.")
    _, h0, k0 = known[0]
    for name, h, k in known[1:]:
        if not math.isclose(float(h), float(h0)) or int(k) != int(k0):
            listing = ", ".join(f"{n}: H={hh}, K={kk}" for n, hh, kk in sources)
            raise ValueError(f"Inconsistent horizon/interval count across sources ({listing}).")
    return float(h0), int(k0)
