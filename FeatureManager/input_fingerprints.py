"""
Fingerprints of every model input source (IMPLEMENTATION_STEPS.md §2.1 lines 72-73,
added 2026-10-08 before the multi-machine runs).

The data version used by the survival labels covers only the outage side
(outage matrices, line list, split indices). This module also identifies the
weather and graph inputs, so `validate` can show that three machines use the
same inputs and `merge` can refuse results built from different ones.

Components:
    outage_csv          SHA-256 of the preprocessed outage CSV
    weather_mappings    SHA-256 of data_mappings.json
    weather_scaler      SHA-256 of scaler.pkl
    weather_hdf5        SAMPLED content fingerprint of weather_processed.hdf5: file
                        size, every dataset's name/shape/dtype, the feature names, and
                        SHA-256 of evenly spaced time slices (default 64 in total).
                        Content-based (independent of file timestamps); a change that
                        misses every sampled slice is not detected (a preprocessing
                        rerun rescales all values and is detected).
    graph_pickle        SHA-256 of model_data/bpa_graph.pkl
    model_tensors       hash of what the model is actually fed: static line features,
                        line ids/xref indices, weather->line and line->line edges and
                        attributes, the time-since-last-outage matrix and (if built)
                        weather->weather edges.

`inputs_fingerprint` is a hash over all components.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

INPUT_FINGERPRINT_VERSION = "input_fingerprints_v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hdf5_sampled_fingerprint(path: str | Path, n_slices: int = 64) -> dict[str, Any]:
    """Sampled content fingerprint of an HDF5 file (see module docstring)."""
    import h5py

    path = Path(path)
    datasets: list[tuple[str, Any]] = []
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: datasets.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
        datasets.sort(key=lambda kv: kv[0])
        features = f.attrs.get("features")
        feature_names = [x.decode() if isinstance(x, bytes) else str(x) for x in features] if features is not None else None
        total_t = sum(int(ds.shape[0]) for _, ds in datasets if ds.ndim >= 1) or 1
        info = []
        for name, ds in datasets:
            info.append({"name": name, "shape": list(ds.shape), "dtype": str(ds.dtype)})
            h.update(f"{name}|{ds.shape}|{ds.dtype}".encode())
            if ds.ndim == 0 or ds.shape[0] == 0:
                continue
            k = max(1, int(round(n_slices * ds.shape[0] / total_t)))
            for i in np.unique(np.linspace(0, ds.shape[0] - 1, k).round().astype(np.int64)):
                h.update(np.ascontiguousarray(ds[int(i)]).tobytes())
    return {"size_bytes": path.stat().st_size, "datasets": info, "feature_names": feature_names,
            "n_slices": n_slices, "sampled_sha256": h.hexdigest()}


def _tensor_hash(arrays: list[tuple[str, Any]]) -> str:
    h = hashlib.sha256()
    for name, a in arrays:
        if a is None:
            h.update(f"{name}:none".encode())
            continue
        arr = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        arr = np.ascontiguousarray(arr)
        h.update(f"{name}:{arr.shape}:{arr.dtype}".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def compute_input_fingerprints(survival_manager, *, graph_pickle_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Fingerprint all inputs behind a SurvivalOutageDataManager (fixtures: missing files -> None)."""
    from FeatureManager.outage_events import file_fingerprint

    base = survival_manager.base
    comps: dict[str, Any] = {}
    csv_path = getattr(getattr(base, "outage_matcher", None), "csv_path", None)
    comps["outage_csv"] = file_fingerprint(csv_path) if csv_path and Path(csv_path).exists() else getattr(base, "outage_csv_fingerprint", None)

    wp = getattr(base, "weather_preprocessor", None)
    hdf5_detail = None
    for key, attr in (("weather_mappings", "hdf5_mappings_path"), ("weather_scaler", "scaler_path")):
        p = getattr(wp, attr, None) if wp is not None else None
        comps[key] = _sha256_file(Path(p)) if p is not None and Path(p).exists() else None
    p = getattr(wp, "processed_hdf5_path", None) if wp is not None else None
    if p is not None and Path(p).exists():
        hdf5_detail = hdf5_sampled_fingerprint(p)
        comps["weather_hdf5"] = hdf5_detail["sampled_sha256"]
    else:
        comps["weather_hdf5"] = None

    gp = graph_pickle_path or getattr(base, "graph_pickle_path", None)
    comps["graph_pickle"] = _sha256_file(Path(gp)) if gp is not None and Path(gp).exists() else None

    m = survival_manager
    comps["model_tensors"] = _tensor_hash([
        ("line_ids", np.array([str(x) for x in m.line_ids])), ("line_xref_idx", m.line_xref_idx),
        ("line_features_static", m.line_features_static),
        ("weather_edges_index", m.weather_edges_index), ("weather_edges_attr", m.weather_edges_attr),
        ("line_edges_index", m.line_edges_index), ("line_edges_attr", m.line_edges_attr),
        ("time_since_matrix", m.time_since_matrix),
        ("weather_weather_edge_index", m.weather_weather_edge_index),
        ("weather_weather_edge_attr", m.weather_weather_edge_attr),
    ])
    payload = json.dumps({"version": INPUT_FINGERPRINT_VERSION, **comps}, sort_keys=True)
    return {
        "inputs_fingerprint": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "version": INPUT_FINGERPRINT_VERSION,
        "components": {k: (v[:16] if isinstance(v, str) else v) for k, v in comps.items()},
        "weather_hdf5_detail": hdf5_detail,
    }


def describe_differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Names of the input components that differ between two fingerprint records."""
    labels = {"outage_csv": "preprocessed outage CSV", "weather_mappings": "weather data_mappings.json",
              "weather_scaler": "weather scaler.pkl", "weather_hdf5": "weather HDF5 (weather_processed.hdf5)",
              "graph_pickle": "graph pickle (bpa_graph.pkl)",
              "model_tensors": "model input tensors (line features / edges / time-since matrix)"}
    ca, cb = (a or {}).get("components", {}), (b or {}).get("components", {})
    return [labels.get(k, k) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)]
