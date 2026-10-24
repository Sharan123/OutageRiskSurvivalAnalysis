"""Preprocessing for the BPA survival outage study: graph, weather, outages, labels.

Run from the repository root, in this order (README.md lists the input data):

    python preprocess.py convert-outages   # Datasets/BPA_xlsx/*.xlsx -> Datasets/BPA_trm/*.csv
    python preprocess.py match-outages     # -> BPA_Line_Outage_Matcher/preprocessed_outages_2018_2023.csv
    python preprocess.py graph             # -> model_data/bpa_graph.pkl + .json
    python preprocess.py weather           # Open-Meteo -> Datasets/Weather/processed/* (needs the graph)
    python preprocess.py labels            # -> FeatureManager/data/processed, SurvivalModels/data/processed

The settings below are those of the training repository (BaselineModels/baseline_utils.py
`build_manager` and the operating_spell_v3 manifest), so the label caches get the same names
there and can be copied across.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

YEARS = [str(y) for y in range(2016, 2024)]   # years the outage matcher reads
GRAPH = ROOT / "model_data" / "bpa_graph.pkl"
OUTAGE_CSV = ROOT / "BPA_Line_Outage_Matcher" / "preprocessed_outages_2018_2023.csv"

# OutageDataManager settings (training repo: BaselineModels/baseline_utils.py)
START_DATE, END_DATE = "2016-01-01", "2023-12-31"
TRAIN_SPLIT, VAL_SPLIT = "2022-01-01", "2023-01-01"
INPUT_WIN = OUTPUT_WIN = 168
NEGATIVE_RATIO = 1000000

# Survival label settings (study operating_spell_v3: trial_65 data_config)
MAX_HORIZON, INTERVALS = 1008, 6
STRIDES = {"train": 24, "val": 24, "test": 1}


def convert_outages(args) -> None:
    from BPA_Line_Outage_Matcher.bpa_xlsx_to_trm_csv import read_line_table, write_csv
    src = Path(args.xlsx_dir)
    out = ROOT / "Datasets" / "BPA_trm"
    out.mkdir(parents=True, exist_ok=True)
    missing = [y for y in YEARS if not (src / f"OutagesCY{y}.xlsx").exists()]
    if missing:
        raise SystemExit(f"missing in {src}: " + ", ".join(f"OutagesCY{y}.xlsx" for y in missing))
    for y in YEARS:
        header, rows = read_line_table(src / f"OutagesCY{y}.xlsx")
        dst = out / f"OutagesCY{y}-TransmissionLineInterruptions.csv"
        write_csv(header, rows, dst)
        print(f"{dst.relative_to(ROOT)}: {len(rows)} rows, {len(header)} columns")


def graph(args) -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        # the graph holds Python sets; the v3 graph was verified with hash seed 0, so rerun with it
        import subprocess
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        raise SystemExit(subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env).returncode)
    from BPA_Spatial_Preprocessing.compare_graphs import compare, rebuild
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    built = rebuild(out)   # stops unless 0 line ends are left free (unmatched)
    print(f"wrote {built} and {built.with_suffix('.json')}")
    if args.compare_with:
        diffs = compare(Path(args.compare_with), built)
        print("GRAPHS_EQUAL (ignoring set order)" if not diffs else "GRAPHS_DIFFER:\n  " + "\n  ".join(diffs))


def weather(args) -> None:
    from BPA_Spatial_Preprocessing.BPAGraphCreator import BPAGraphCreator
    from WeatherPreprocessor.WeatherPreprocessor import WeatherPreprocessor
    if not GRAPH.exists():
        raise SystemExit(f"missing {GRAPH}: run `python preprocess.py graph` first")
    wp = WeatherPreprocessor(BPAGraphCreator.load(str(GRAPH)), start_date=START_DATE, end_date=END_DATE,
                             verbose=False, buffer_km=30.0)
    steps = args.steps or ["download", "raw", "clean", "transform"]
    if "download" in steps:   # skips cell files that already exist, so it can be rerun after an interruption
        wp.obtain_partially(api_key=args.api_key or os.environ.get("OPEN_METEO_API_KEY"))
    if "raw" in steps:
        wp.create_raw_hdf5(max_workers=args.workers)
    if "clean" in steps:
        wp.clean_dataset(temporal_limit=2, num_workers=args.workers)
    if "transform" in steps:
        wp.transform_ds(scaler_type="minmax", train_split=TRAIN_SPLIT, valid_split=VAL_SPLIT, num_workers=args.workers)


def match_outages(args) -> None:
    from BPA_Line_Outage_Matcher.BPA_Outage_Line_Matcher import BPAOutageMatcher
    matcher = BPAOutageMatcher(years=YEARS, verbose=args.verbose)
    matcher.check_validity()
    matcher.preprocess()
    matcher.print_unmatched_outage_names()
    print("\nBPA times Pacific -> UTC, per source year:")
    for year, audit in matcher.timezone_audit.items():
        print(f"  {year}: {audit}")
    matcher.save_csv(filename=str(OUTAGE_CSV))
    print(f"\nwrote {OUTAGE_CSV.relative_to(ROOT)}")


def build_base_manager():
    from BPA_Line_Outage_Matcher.BPA_Outage_Line_Matcher import BPAOutageMatcher
    from BPA_Spatial_Preprocessing.BPAGraphCreator import BPAGraphCreator
    from FeatureManager.OutageDataManager import OutageDataManager
    from WeatherPreprocessor.WeatherPreprocessor import WeatherPreprocessor
    for p in (GRAPH, OUTAGE_CSV):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    graph = BPAGraphCreator.load(str(GRAPH))
    if not graph.cell_coverage_cache:
        graph.get_cells_coverage()
    wp = WeatherPreprocessor(graph=graph, start_date=START_DATE, end_date=END_DATE)
    wp._load_mappings_if_needed()
    matcher = BPAOutageMatcher(verbose=False)
    matcher.load_csv()
    manager = OutageDataManager(
        graph_creator=graph, weather_preprocessor=wp, outage_matcher=matcher, start_date=START_DATE,
        end_date=END_DATE, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT, input_window_hours=INPUT_WIN,
        output_window_hours=OUTPUT_WIN, tiny_mode=False, processed_dir=ROOT / "FeatureManager" / "data" / "processed",
        verbose=False, negative_ratio=NEGATIVE_RATIO, valid_stride=1,
    )
    manager.graph_pickle_path = GRAPH   # fingerprinted by FeatureManager/input_fingerprints.py
    return manager


def labels(args) -> None:
    from FeatureManager.input_fingerprints import compute_input_fingerprints
    from SurvivalModels.SurvivalOutageDataManager import SurvivalOutageDataManager
    base = build_base_manager()
    m = SurvivalOutageDataManager(base, max_horizon=MAX_HORIZON, discrete_intervals=INTERVALS,
                                  train_stride=STRIDES["train"], val_stride=STRIDES["val"], test_stride=STRIDES["test"],
                                  verbose=True, processed_dir=str(ROOT / "SurvivalModels" / "data" / "processed"))
    fp = compute_input_fingerprints(m)
    summary = {
        "outage_cache": str(Path(base.meta_path).relative_to(ROOT)),
        "label_cache": str(Path(m.meta_path).relative_to(ROOT)),
        "outage_csv_fingerprint": base.outage_csv_fingerprint,
        "inputs_fingerprint": fp.get("inputs_fingerprint"),
        "input_components": fp.get("components"),
        "audit": m.data_audit(),
    }
    out = ROOT / "SurvivalModels" / "data" / "processed" / "preprocessing_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\noutage cache: {summary['outage_cache']}\nlabel cache:  {summary['label_cache']}")
    print(f"inputs_fingerprint: {summary['inputs_fingerprint']}")
    for k, v in (summary["input_components"] or {}).items():
        print(f"  {k:18s} {v}")
    print(f"summary: {out.relative_to(ROOT)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("convert-outages", help="BPA xlsx -> Datasets/BPA_trm CSVs (2016-2023)")
    p.add_argument("--xlsx-dir", default=str(ROOT / "Datasets" / "BPA_xlsx"))
    p.set_defaults(func=convert_outages)
    p = sub.add_parser("match-outages", help="match outages to graph lines, Pacific -> UTC, write the CSV")
    p.add_argument("--verbose", action="store_true", help="matcher verbose output (very long)")
    p.set_defaults(func=match_outages)
    p = sub.add_parser("graph", help="build the line graph (fails unless 0 free line ends)")
    p.add_argument("--out-dir", default=str(ROOT / "model_data"))
    p.add_argument("--compare-with", help="an existing bpa_graph.pkl to compare with, ignoring set order")
    p.set_defaults(func=graph)
    p = sub.add_parser("weather", help="download and process Open-Meteo weather for the graph's cells")
    p.add_argument("--steps", nargs="+", choices=["download", "raw", "clean", "transform"])
    p.add_argument("--api-key", help="Open-Meteo customer key (or env OPEN_METEO_API_KEY); default: free API")
    p.add_argument("--workers", type=int, help="processes for the HDF5 steps (default: all cores)")
    p.set_defaults(func=weather)
    p = sub.add_parser("labels", help="build the outage matrices and survival labels (cached)")
    p.set_defaults(func=labels)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
