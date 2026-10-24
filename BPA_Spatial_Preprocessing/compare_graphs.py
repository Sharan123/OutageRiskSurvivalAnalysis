"""Rebuild the BPA line graph and/or compare two saved graphs, ignoring set order.

The graph (model_data/bpa_graph.pkl + .json) is built only from the line GeoJSON and the rule
files in this folder; it does not read outage data. Python sets inside it make the pickle bytes
depend on PYTHONHASHSEED, so equality is checked on content.

    PYTHONHASHSEED=0 python BPA_Spatial_Preprocessing/compare_graphs.py --rebuild <dir>   # build into <dir>
    python BPA_Spatial_Preprocessing/compare_graphs.py model_data/bpa_graph.pkl <dir>/bpa_graph.pkl

`--rebuild` runs the notebook steps (WeatherPreprocessor/TestingWeatherIntegration.ipynb cells
74-78) and fails unless 0 line ends are left free (unmatched).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rebuild(out_dir: Path) -> Path:
    from BPA_Spatial_Preprocessing.BPAGraphCreator import BPAGraphCreator
    g = BPAGraphCreator()
    g.getLineStringIntersects()
    g.assignSubstationToRemainingFreePoint()
    g.match_proximity_substations()
    free = g.line_ends_dict.total_num_free_points()
    print(f"free (unmatched) line ends: {free}")
    if free != 0:
        raise SystemExit(f"{free} free line ends remain")
    out = out_dir / "bpa_graph.pkl"
    g.save(out)
    return out


def _end_key(e) -> tuple:
    return (e.free_point, e.connection_type.name, e.substation_name, tuple(sorted(e.all_merging_lines_xrefcd)),
            e.point.wkb, e.associated_line_str.wkb, e.line_idx, e.end.name)


def _ends(v):
    """mapping_dict value: a [start, end] pair, or a list of pairs for multi-part lines."""
    if v and isinstance(v[0], (list, tuple)):
        return [tuple(_end_key(e) for e in pair) for pair in v]
    return [tuple(_end_key(e) for e in v)]


def _json_norm(info: dict) -> dict:
    out = {}
    for k, d in info.items():
        d = dict(d)
        if "line_connections" in d:
            d["line_connections"] = sorted(d["line_connections"])
        if "substation_connections" in d:
            d["substation_connections"] = sorted(
                ({s: sorted(v) for s, v in item.items()} for item in d["substation_connections"]),
                key=lambda x: json.dumps(x, sort_keys=True))
        out[k] = d
    return out


def compare(a: Path, b: Path) -> list[str]:
    pa, pb = (pickle.loads(p.read_bytes()) for p in (a, b))
    diffs = []
    if not pa["lines"].equals(pb["lines"]):
        diffs.append("lines GeoDataFrame differs")
    la, lb = pa["line_ends_dict"], pb["line_ends_dict"]
    ma, mb = la["mapping_dict"], lb["mapping_dict"]
    if set(ma) != set(mb):
        diffs.append(f"mapping_dict keys differ: {sorted(set(ma) ^ set(mb))[:10]}")
    diffs += [f"line ends differ for {k}" for k in sorted(set(ma) & set(mb)) if _ends(ma[k]) != _ends(mb[k])][:20]
    sa = {k: sorted(v) for k, v in la["line_subs_dict"].items()}
    sb = {k: sorted(v) for k, v in lb["line_subs_dict"].items()}
    if sa != sb:
        diffs.append("line_subs_dict differs")
    ja, jb = (json.loads(p.with_suffix(".json").read_text()) for p in (a, b))
    if list(ja) != list(jb):
        diffs.append("JSON key order differs (line ids / edge order)")
    if _json_norm(ja) != _json_norm(jb):
        diffs.append("JSON content differs")
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="*", help="two .pkl files to compare")
    ap.add_argument("--rebuild", type=Path, help="rebuild into this directory and compare with model_data/")
    args = ap.parse_args()
    if args.rebuild:
        args.rebuild.mkdir(parents=True, exist_ok=True)
        args.graphs = [str(ROOT / "model_data" / "bpa_graph.pkl"), str(rebuild(args.rebuild))]
    if len(args.graphs) != 2:
        ap.error("give two graphs or --rebuild")
    diffs = compare(Path(args.graphs[0]), Path(args.graphs[1]))
    print("GRAPHS_EQUAL (ignoring set order)" if not diffs else "GRAPHS_DIFFER:\n  " + "\n  ".join(diffs))
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
