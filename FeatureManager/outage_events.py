"""
Normalised outage records shared by the outage matcher, the binary outage
matrices and the survival event registry (IMPLEMENTATION_STEPS.md §3.4).

Pipeline (each step is a function below):

  1. `consolidate_matched_outages`  (called by BPA_Outage_Line_Matcher.preprocess)
        one entry per (Outage ID, XRefCd); outage-level cause rule; latest known
        restore time. Replaces the old group-by-cause + keep-first-alphabetical
        dedup, which silently dropped rows (e.g. the Weather rows of 200342).
  2. `iter_normalized_outage_records` / `build_event_registry`
        hour rounding exactly as before; every record kept with a mapping status.
  3. same-hour episode merge (inside `build_event_registry`)
        records on the same line whose onset falls in the same clock hour are
        one line episode: ID = earliest Outage ID (all IDs kept), restore = latest
        KNOWN restore time ("still out" only if none is known), weather if any
        record is weather.
  4. `matrices_from_registry`
        binary matrices built from the merged episodes, so matrices, labels and
        event IDs share one code path.

Kept free of heavy imports (shapely/h5py/graph code) so tests can use it.
"""

from __future__ import annotations

from datetime import timedelta
import hashlib
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

# Bump when consolidation / rounding / episode-merge rules change. Part of the
# feature-cache key in OutageDataManager.
OUTAGE_EVENTS_VERSION = "outage_events_v3"  # v3: BPA Pacific times converted to UTC

# Weather-related causes defining the event (label contract v2, decided 2026-10-08).
# 'Tree blown' and 'Tree Blown' are both spellings present in the BPA files.
# Galloping Conductors: wind-driven oscillation of iced conductors (ice + wind).
# Lightning: included although no lightning data is available as an input.
# Possible future expansions (NOT included): 'Fire' (wildfire, but can be human
# caused), 'Smoke' (wildfire-smoke flashover), 'Tree' (needs vegetation data).
DEFAULT_WEATHER_CAUSES = ("Ice", "Tree blown", "Tree Blown", "Weather", "Wind", "Lightning", "Galloping Conductors")

UNKNOWN_CAUSES = {"unknown", "not reported", "nan", ""}

# mapping_status values
MAPPED = "mapped"
LINE_NOT_IN_GRAPH = "line_not_in_graph"
STARTS_AFTER_RANGE = "starts_after_range"
START_NOT_ON_GRID = "start_not_on_hour_grid"
XREF_END_MISMATCH = "xref_end_count_mismatch"
END_MISSING = "recovery_time_missing"
END_UNPARSEABLE = "recovery_time_unparseable"
ENDED_BEFORE_RANGE = "ended_before_range"
NONPOSITIVE_DURATION = "nonpositive_rounded_duration"

NAT_FLAG = "recovery_NaT_extended_to_end_of_data"


def file_fingerprint(path: str | Path) -> str:
    """SHA-256 of a file (used to key caches on the preprocessed outage CSV)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_outage_id(value) -> Optional[str]:
    """'196240.0' / 196240.0 / 196240 -> '196240'; other strings kept verbatim."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def _ceil_hour(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        return ts
    return ts.floor("h") + timedelta(hours=1)


def _is_unknown(cause) -> bool:
    return cause is None or (isinstance(cause, float) and np.isnan(cause)) or str(cause).strip().lower() in UNKNOWN_CAUSES


# ============================================================================ #
#  0. Source timezone: BPA Pacific wall-clock time -> UTC (R5.2, label v3)       #
# ============================================================================ #

BPA_SOURCE_TIMEZONE = "America/Los_Angeles"
BPA_TIME_FORMAT = "%m/%d/%Y %H:%M"


def _localize(local: pd.Series, dst_flags: np.ndarray) -> pd.Series:
    """Localize naive Pacific times; `dst_flags` resolves the repeated fall-back hour
    (True = first occurrence, PDT). Nonexistent spring-forward times move forward."""
    return local.dt.tz_localize(BPA_SOURCE_TIMEZONE, ambiguous=dst_flags, nonexistent="shift_forward")


def bpa_pacific_to_utc(out_raw: pd.Series, in_raw: pd.Series, duration_raw: pd.Series,
                       outage_id: Optional[pd.Series] = None) -> tuple[pd.Series, pd.Series, dict]:
    """Convert BPA `Out`/`In Datetime` strings (Pacific wall clock, PST/PDT) to tz-naive UTC.

    BPA publishes local times (the 2022+ files label them "(PPT)"); the weather index is UTC.
    Repeated fall-back hour: the reading whose (In - Out) matches BPA's `Duration (minutes)`
    within 1 minute is used; with no In/Duration, or no matching reading, the first
    occurrence (PDT). The choice for an Out time is shared by all rows of the same Outage ID
    with the same raw Out string. Spring-forward times that do not exist are shifted forward.
    Returns (out_utc, in_utc, audit counts).
    """
    out_local = pd.to_datetime(out_raw, format=BPA_TIME_FORMAT, errors="coerce")
    in_local = pd.to_datetime(in_raw.where(in_raw.astype(str).str.strip() != ""), format=BPA_TIME_FORMAT, errors="coerce")
    bad_out = out_raw.notna() & out_local.isna()
    if bad_out.any():
        raise ValueError(f"unparseable Out Datetime values: {out_raw[bad_out].unique()[:5].tolist()}")
    dur = pd.to_numeric(duration_raw, errors="coerce")
    n = len(out_local)
    t, f = np.ones(n, bool), np.zeros(n, bool)

    def amb(local: pd.Series) -> np.ndarray:
        a, b = _localize(local, t), _localize(local, f)
        return (local.notna() & (a != b)).to_numpy()  # NaT != NaT is True: exclude missing times

    def nonexistent(local: pd.Series) -> np.ndarray:
        shifted = _localize(local, t).dt.tz_localize(None)  # local wall clock after shifting
        return (local.notna() & (shifted != local)).to_numpy() & ~amb(local)

    out_amb, in_amb = amb(out_local), amb(in_local)
    out_dst, in_dst = t.copy(), t.copy()
    resolved = np.zeros(n, bool)
    for i in np.flatnonzero(out_amb | in_amb):
        if pd.isna(in_local.iloc[i]) or pd.isna(dur.iloc[i]):
            continue
        best = None
        for od in ((True, False) if out_amb[i] else (True,)):
            for idd in ((True, False) if in_amb[i] else (True,)):
                o = out_local.iloc[i].tz_localize(BPA_SOURCE_TIMEZONE, ambiguous=od)
                e = in_local.iloc[i].tz_localize(BPA_SOURCE_TIMEZONE, ambiguous=idd)
                if abs((e - o) / pd.Timedelta(minutes=1) - dur.iloc[i]) <= 1 and best is None:
                    best = (od, idd)
        if best is not None:
            out_dst[i], in_dst[i] = best
            resolved[i] = True
    conflicts = 0
    if outage_id is not None and out_amb.any():
        # one Out reading per (Outage ID, raw Out): taken from the first duration-resolved row
        key = pd.DataFrame({"id": outage_id.to_numpy(), "raw": out_raw.to_numpy(), "dst": out_dst,
                            "res": resolved}).iloc[np.flatnonzero(out_amb)]
        for _, g in key.groupby(["id", "raw"]):
            decided = g.loc[g["res"], "dst"]
            if decided.nunique() > 1:
                conflicts += 1
            out_dst[g.index.to_numpy()] = bool(decided.iloc[0]) if len(decided) else True
    out_utc = _localize(out_local, out_dst).dt.tz_convert("UTC").dt.tz_localize(None)
    in_utc = _localize(in_local, in_dst).dt.tz_convert("UTC").dt.tz_localize(None)
    audit = {"rows": int(n), "ambiguous_out": int(out_amb.sum()), "ambiguous_in": int(in_amb.sum()),
             "ambiguous_resolved_by_duration": int(resolved.sum()), "ambiguous_id_conflicts": int(conflicts),
             "nonexistent_out": int(nonexistent(out_local).sum()), "nonexistent_in": int(nonexistent(in_local).sum())}
    return out_utc, in_utc, audit


# ============================================================================ #
#  1. Matcher-level consolidation                                              #
# ============================================================================ #

def dedupe_hits(hits: Sequence, types: Sequence) -> tuple[list, list]:
    """Remove repeated XRefCds from one match result, keeping order (and types aligned).

    The line inventory has two name rows for some codes (e.g. HALNT-ALEU-1), so a
    fuzzy match used to add the same code twice.
    """
    seen, out_h, out_t = set(), [], []
    for h, t in zip(hits, types):
        if h in seen:
            continue
        seen.add(h)
        out_h.append(h)
        out_t.append(t)
    return out_h, out_t


def outage_cause(causes: Sequence, weather_causes: Iterable[str] = DEFAULT_WEATHER_CAUSES) -> str:
    """Outage-level cause rule (decided 2026-10-08):
    a weather cause if any row has one (weather wins over other known causes),
    else the first non-Unknown cause, else the first source wording (e.g.
    'Not Reported') or 'Unknown'. 'First' = source row order."""
    weather = set(weather_causes)
    for c in causes:
        if c in weather:
            return c
    for c in causes:
        if not _is_unknown(c):
            return c
    # no known cause: keep the source wording (e.g. 'Not Reported') if there is one
    return next((str(c) for c in causes if not (isinstance(c, float) and np.isnan(c)) and str(c).strip()), "Unknown")


CSV_COLUMNS = ["Outage ID", "Voltage (kV)", "Out Datetime", "Cause", "Responsible System", "In Datetime",
               "Original Line Name", "Line Type", "O&M District", "Length (miles)", "Duration (minutes)", "Name",
               "XRefCd", "XRefCd_Type", "Source Causes", "Source Rows",
               "Out Datetime (Pacific)", "In Datetime (Pacific)"]  # raw BPA wall-clock audit columns (v3)


def consolidate_matched_outages(combined_with_lines: pd.DataFrame,
                                weather_causes: Iterable[str] = DEFAULT_WEATHER_CAUSES) -> pd.DataFrame:
    """One row per Outage ID with one sub-event per UNIQUE XRefCd.

    Input: the matcher's `combined_with_lines` (one source row per line record,
    `XRefCd` / `XRefCd_Type` = list of matched codes). Output: the
    preprocessed-CSV schema (list columns aligned to sub-events) plus two audit
    columns, `Source Causes` and `Source Rows`.

      * Cause: `outage_cause` over all rows of the outage.
      * Start: earliest `Out Datetime` of the outage.
      * Per XRefCd: restore = latest KNOWN `In Datetime` among its rows; NaT
        ("still out") only if no row has a time. Descriptive fields come from
        the first row that matched the code.
    """
    rows = []
    for oid, g in combined_with_lines.groupby("Outage ID", sort=True):
        causes = list(g["Cause"])
        cause = outage_cause(causes, weather_causes)
        out_all = pd.to_datetime(g["Out Datetime"])
        out_dt = out_all.min()
        has_pac = "Out Datetime (Pacific)" in g.columns
        out_pac = str(g["Out Datetime (Pacific)"].loc[out_all.idxmin()]) if has_pac else ""
        resp = next((r for r in g["Responsible System"] if not _is_unknown(r)), g["Responsible System"].iloc[0])
        per_x: dict[str, dict] = {}
        for _, r in g.iterrows():
            xs, ts = dedupe_hits(list(r["XRefCd"]), list(r["XRefCd_Type"]))
            for x, t in zip(xs, ts):
                e = per_x.setdefault(x, {"type": t, "orig": r["Original Line Name"], "ltype": r["Line Type"],
                                         "district": r["O&M District"], "length": r["Length (miles)"],
                                         "name": r["Name"], "ins": []})
                raw_in = r["In Datetime (Pacific)"] if has_pac else ""
                e["ins"].append((pd.to_datetime(r["In Datetime"]), "" if pd.isna(raw_in) else str(raw_in)))
        ins, ins_pac, durs = [], [], []
        for e in per_x.values():
            known = [(t, raw) for t, raw in e["ins"] if not pd.isna(t)]
            restore, restore_pac = max(known, key=lambda k: k[0]) if known else (pd.NaT, "NaT")
            ins.append(restore)
            ins_pac.append(restore_pac or "NaT")
            durs.append("still out" if pd.isna(restore) else str(int(round((restore - out_dt) / pd.Timedelta(minutes=1)))))
        rows.append({
            "Outage ID": oid, "Voltage (kV)": g["Voltage (kV)"].iloc[0], "Out Datetime": out_dt, "Cause": cause,
            "Responsible System": resp, "In Datetime": ins,
            "Original Line Name": [e["orig"] for e in per_x.values()], "Line Type": [e["ltype"] for e in per_x.values()],
            "O&M District": [e["district"] for e in per_x.values()], "Length (miles)": [e["length"] for e in per_x.values()],
            "Duration (minutes)": durs, "Name": [e["name"] for e in per_x.values()],
            "XRefCd": [[x] for x in per_x], "XRefCd_Type": [[e["type"]] for e in per_x.values()],
            "Source Causes": "|".join(str(c) for c in causes), "Source Rows": int(len(g)),
            "Out Datetime (Pacific)": out_pac, "In Datetime (Pacific)": ins_pac,
        })
    out = pd.DataFrame(rows, columns=CSV_COLUMNS)
    assert not out["Outage ID"].duplicated().any()
    return out


# ============================================================================ #
#  2. Record normalisation (hour rounding)                                     #
# ============================================================================ #

def iter_normalized_outage_records(
    grouped: pd.DataFrame,
    line_ids: Sequence[str],
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    weather_causes: Iterable[str] = DEFAULT_WEATHER_CAUSES,
) -> Iterator[dict]:
    """Yield normalised records, one per (outage row, sub-event, line code).

    Rounding/skip rules are those of the original `_process_outages_and_timeline`:
      * start floored to the hour; a start before `start_date` is clamped to 0
      * recovery ceiled to the hour; recovery after the range -> len(timeline)
      * a recovery that is NaT ("still out") is flagged and, unless the same-hour
        episode merge finds a known restore time, extends to the end of the data
      * rows with mismatched XRefCd/recovery list lengths are skipped entirely
    Records that would not touch the matrices are still yielded with a
    non-`mapped` status so the audit table can count them.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    effective_end = end_date + pd.Timedelta(hours=23)
    n_hours = int((effective_end - start_date) / pd.Timedelta(hours=1)) + 1
    line_to_idx = {x: i for i, x in enumerate(line_ids)}
    weather_causes = set(weather_causes)

    def hour_index(ts: pd.Timestamp) -> Optional[int]:
        delta = ts - start_date
        hours = delta / pd.Timedelta(hours=1)
        if hours != int(hours) or hours < 0 or hours >= n_hours:
            return None
        return int(hours)

    for row_pos, (_, row) in enumerate(grouped.iterrows()):
        outage_id = row.get("Outage ID")
        cause = row.get("Cause")
        is_weather = cause in weather_causes
        raw_start = pd.to_datetime(row["Out Datetime"])
        start_time = raw_start.floor("h")
        xref_groups = row["XRefCd"]
        end_times = row["In Datetime"]

        base = {
            "source_outage_id": _normalize_outage_id(outage_id),
            "source_row": int(row_pos),
            "cause": None if pd.isna(cause) else str(cause),
            "weather_cause": bool(is_weather),
            "raw_start_time": raw_start,
            "rounded_start_time": start_time,
        }

        status_row: Optional[str] = None
        if start_time > effective_end:
            status_row = STARTS_AFTER_RANGE
            t_start = None
        elif start_time < start_date:
            t_start = 0
        else:
            t_start = hour_index(start_time)
            if t_start is None:
                status_row = START_NOT_ON_GRID
        if status_row is None and len(xref_groups) != len(end_times):
            status_row = XREF_END_MISMATCH

        if status_row is not None:
            for sub_idx, involved in enumerate(xref_groups):
                for line_code in involved:
                    yield {
                        **base, "sub_event": sub_idx, "XRefCd": str(line_code),
                        "line_idx": line_to_idx.get(line_code, -1), "raw_recovery_time": None,
                        "rounded_recovery_time": None, "t_start": t_start, "t_end": None,
                        "mapping_status": status_row, "data_quality_flag": None,
                    }
            continue

        for sub_idx, (involved, end_str) in enumerate(zip(xref_groups, end_times)):
            status = MAPPED
            quality_flag = None
            raw_end = None
            end_time = None
            t_end = None
            try:
                if pd.isna(end_str) or str(end_str).lower() == "nan":
                    status = END_MISSING
                else:
                    raw_end = pd.to_datetime(end_str)
            except Exception:
                status = END_UNPARSEABLE
            if status == MAPPED and pd.isna(raw_end):
                # "still out" in the BPA files: open when the yearly file was
                # compiled. Treated as unknown recovery -> out to end of data,
                # unless a same-hour record on the line supplies a restore time.
                quality_flag = NAT_FLAG
                t_end = n_hours
                if t_end <= t_start:
                    status = NONPOSITIVE_DURATION
            elif status == MAPPED:
                end_time = _ceil_hour(raw_end)
                if end_time <= start_date:
                    status = ENDED_BEFORE_RANGE
                else:
                    if end_time > effective_end:
                        t_end = n_hours
                    else:
                        t_end = hour_index(end_time)
                        if t_end is None:
                            t_end = n_hours
                    if t_end <= t_start:
                        status = NONPOSITIVE_DURATION
            for line_code in involved:
                idx = line_to_idx.get(line_code, -1)
                line_status = status
                if line_status == MAPPED and idx < 0:
                    line_status = LINE_NOT_IN_GRAPH
                yield {
                    **base, "sub_event": sub_idx, "XRefCd": str(line_code), "line_idx": int(idx),
                    "raw_recovery_time": raw_end, "rounded_recovery_time": end_time, "t_start": int(t_start),
                    "t_end": None if t_end is None else int(t_end), "mapping_status": line_status,
                    "data_quality_flag": quality_flag,
                }


# ============================================================================ #
#  3. Registry with same-hour episode merge                                    #
# ============================================================================ #

def build_event_registry(
    grouped: pd.DataFrame,
    line_ids: Sequence[str],
    *,
    start_date,
    end_date,
    weather_causes: Iterable[str] = DEFAULT_WEATHER_CAUSES,
) -> pd.DataFrame:
    """Record-level registry with merged line-episode columns.

    Record columns: source_outage_id, incident_episode_id (= INC-<own Outage ID>),
    source_record_key, raw/rounded times, t_start/t_end, mapping_status,
    data_quality_flag.

    Episode columns (same for every record of one line episode):
      line_episode_id      LEP-<XRefCd>-<t_start>; all mapped records of a line
                           whose onset is in the same clock hour
      episode_primary      True for the earliest record (earliest raw start, then
                           lowest Outage ID); its incident is the episode's incident
      episode_outage_ids   all Outage IDs in the episode, '|'-joined
      episode_t_end        latest KNOWN rounded restore hour; end of data only if
                           no record knows it (episode_restore_status)
      episode_weather      any record has a weather cause
      duplicate_of         for non-primary records: the primary's record key
    """
    recs = list(iter_normalized_outage_records(grouped, line_ids, start_date=start_date, end_date=end_date,
                                               weather_causes=weather_causes))
    columns = [
        "source_outage_id", "source_row", "sub_event", "cause", "weather_cause", "XRefCd", "line_idx",
        "raw_start_time", "raw_recovery_time", "rounded_start_time", "rounded_recovery_time",
        "t_start", "t_end", "mapping_status", "data_quality_flag",
    ]
    reg = pd.DataFrame(recs, columns=columns)
    extra = ("incident_episode_id", "line_episode_id", "source_record_key", "duplicate_of", "episode_primary",
             "episode_outage_ids", "episode_t_end", "episode_restore_status", "episode_weather", "episode_n_records",
             "episode_causes")
    if reg.empty:
        for c in extra:
            reg[c] = pd.Series(dtype=object)
        reg["registry_idx"] = pd.Series(dtype=np.int64)
        return reg

    reg["incident_episode_id"] = "INC-" + reg["source_outage_id"].astype(str)
    reg["source_record_key"] = (
        "row" + reg["source_row"].astype(str) + "/sub" + reg["sub_event"].astype(str) + "/" + reg["XRefCd"]
    )
    reg["registry_idx"] = np.arange(len(reg), dtype=np.int64)
    mapped = reg["mapping_status"] == MAPPED
    t_start_str = reg["t_start"].astype("Int64").astype(str)
    reg["line_episode_id"] = np.where(mapped, "LEP-" + reg["XRefCd"] + "-" + t_start_str,
                                      "UNMAPPED-" + reg["source_record_key"])
    oid_num = pd.to_numeric(reg["source_outage_id"], errors="coerce")
    order = reg.assign(_oid=oid_num).sort_values(["line_episode_id", "raw_start_time", "_oid", "source_row"], kind="stable")
    primary_key = order.groupby("line_episode_id")["source_record_key"].first()
    reg["episode_primary"] = reg["source_record_key"].to_numpy() == primary_key.reindex(reg["line_episode_id"]).to_numpy()
    reg["duplicate_of"] = np.where(reg["episode_primary"], None, primary_key.reindex(reg["line_episode_id"]).to_numpy())
    g = reg.groupby("line_episode_id")
    reg["episode_outage_ids"] = g["source_outage_id"].transform(
        lambda s: "|".join(sorted(set(map(str, s)), key=lambda x: (len(x), x))))
    known_end = reg["t_end"].where(reg["data_quality_flag"].isna())
    ep_known = known_end.groupby(reg["line_episode_id"]).transform("max")
    ep_any = reg["t_end"].groupby(reg["line_episode_id"]).transform("max")
    reg["episode_t_end"] = ep_known.fillna(ep_any)
    reg["episode_restore_status"] = np.where(ep_known.notna(), "known",
                                             np.where(mapped, "still_out_extended_to_end_of_data", None))
    reg["episode_weather"] = g["weather_cause"].transform("any")
    reg["episode_n_records"] = g["source_record_key"].transform("size")
    reg["episode_causes"] = g["cause"].transform(lambda s: "|".join(sorted(set(map(str, s)))))
    return reg


def matrices_from_registry(registry: pd.DataFrame, n_hours: int, n_lines: int) -> tuple[np.ndarray, np.ndarray]:
    """(weather_matrix, all_outages_matrix) built from merged line episodes.

    Each mapped episode marks [t_start, episode_t_end) as out; if any of its
    records is weather, the whole episode is a weather outage.
    """
    om = np.zeros((n_hours, n_lines), dtype=np.int8)
    aom = np.zeros((n_hours, n_lines), dtype=np.int8)
    ep = registry[(registry["mapping_status"] == MAPPED) & registry["episode_primary"].astype(bool)]
    for t0, t1, li, w in zip(ep["t_start"], ep["episode_t_end"], ep["line_idx"], ep["episode_weather"]):
        aom[int(t0):int(t1), int(li)] = 1
        if w:
            om[int(t0):int(t1), int(li)] = 1
    return om, aom
