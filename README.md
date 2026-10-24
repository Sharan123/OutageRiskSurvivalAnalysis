# Survival outage analysis: preprocessing

This repository holds only the **preprocessing** of the BPA transmission-line outage study. It
builds the inputs that the training repository (`GNN_Outage_Prediction`) reads:

1. the yearly BPA outage tables;
2. the outages matched to transmission lines, with BPA's Pacific times converted to UTC;
3. the line graph;
4. the hourly Open-Meteo weather for the cells around every line;
5. the outage matrices and survival labels (operating-spell label contract v3).

Model training, evaluation and the experiment runner are not part of this repository.

All commands run from the repository root through `preprocess.py`. It uses the same settings as
the training repository, so the cache files have the same names there.

## 1. Environment

Python 3.10, with the pinned versions in `requirements.txt` (the same versions as training):
```bash
conda create -n survival_prep python=3.10.16 -y
conda activate survival_prep
python -m pip install -r requirements.txt
```
A GPU is not needed. `torch` and `torch-geometric` are required because the label step builds the
same graph tensors that training uses.

## 2. Data to download and where to put it

| Data | Source | Put it in | Size |
|---|---|---|---|
| BPA outage workbooks, **2016–2023** (8 files) | `https://transmission.bpa.gov/Business/Operations/Outages/OutagesCY<year>.xlsx`, e.g. [OutagesCY2022.xlsx](https://transmission.bpa.gov/Business/Operations/Outages/OutagesCY2022.xlsx) | `Datasets/BPA_xlsx/OutagesCY2016.xlsx` … `OutagesCY2023.xlsx` | small |
| BPA transmission lines | already in the repository | `Datasets/TransmissionLineData/BPA_TransmissionLines.geojson` and `.csv` | – |
| Graph rule files | already in the repository | `BPA_Spatial_Preprocessing/*_rules.txt`, `BPA_Line_Outage_Matcher/rename_name_rules.txt` | – |
| Hourly weather, 2016–2023 | downloaded by `preprocess.py weather` from the [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api) | `Datasets/Weather/open-metro-cells/` (raw CSVs), then `Datasets/Weather/processed/` | about 35 GB raw CSV, 6.5 GB `weather_raw.hdf5`, 11.4 GB `weather_processed.hdf5` |

Use the **xlsx** workbooks, not the HTML pages. The converter reproduces the older 2016–2020 CSVs
byte for byte. Download all eight years again, because BPA revises past years:
- the 2021 CSV previously in this repository was a copy of 2022;
- the earlier 2022 and 2023 files were incomplete.

## 3. Commands, in order

```bash
python preprocess.py convert-outages
python preprocess.py match-outages
python preprocess.py graph
python preprocess.py weather
python preprocess.py labels
```

| Step | Reads | Writes | Check |
|---|---|---|---|
| `convert-outages` | `Datasets/BPA_xlsx/OutagesCY<year>.xlsx` | `Datasets/BPA_trm/OutagesCY<year>-TransmissionLineInterruptions.csv` (2016–2023) | prints rows per year: 2021 has 4,974, 2022 has 2,157, 2023 has 1,743 |
| `match-outages` | the yearly CSVs, the line CSV, the rename rules | `BPA_Line_Outage_Matcher/preprocessed_outages_2018_2023.csv` | prints the unmatched outage names and the Pacific→UTC audit per year |
| `graph` | the line GeoJSON and the rule files | `model_data/bpa_graph.pkl` and `bpa_graph.json` | stops unless **0** line ends are left unmatched |
| `weather` | `model_data/bpa_graph.pkl` | `Datasets/Weather/open-metro-cells/all_cells_gmt0_<i>.csv`, then `Datasets/Weather/processed/{weather_raw.hdf5, weather_processed.hdf5, data_mappings.json, scaler.pkl}` | – |
| `labels` | everything above | `FeatureManager/data/processed/dataset_meta_<hash>.pkl`, `SurvivalModels/data/processed/survival_labels_<hash>.pkl`, `SurvivalModels/data/processed/preprocessing_summary.json` | prints the cache names and the input fingerprints |

**Step details:**
- **`convert-outages`:** `--xlsx-dir <folder>` reads the workbooks from another folder.
- **`match-outages`:** BPA times are Pacific wall-clock time and are converted to UTC here.
  - In the repeated fall-back hour, the reading that matches BPA's `Duration` is chosen.
  - Times that don't exist (the spring-forward hour) are shifted forward.
  - The original times are kept in the `... (Pacific)` columns.
- **`graph`:**
  - It reruns itself with `PYTHONHASHSEED=0`, the setting the v3 graph was checked with.
  - `--compare-with <bpa_graph.pkl>` compares the result with an existing graph, ignoring set order.
  - `--out-dir <folder>` writes somewhere other than `model_data/`.
- **`weather`** runs four sub-steps: `download`, `raw`, `clean` and `transform`.
  - Run some of them with `--steps download`, or e.g. `--steps raw clean transform`.
  - The download skips files it already has, so after an interruption just run it again.
  - `--api-key` (or the environment variable `OPEN_METEO_API_KEY`) uses the Open-Meteo customer
    endpoint. Without a key the free endpoint is used, which has rate limits.
  - `--workers N` limits the number of processes for the HDF5 sub-steps.
  - Splits are train < 2022-01-01 ≤ validation < 2023-01-01 ≤ test. The min-max scaler is fitted
    on train only.
- **`labels`** uses the study settings:
  - window 168 h;
  - horizon 1,008 h (6 weeks) in 6 intervals;
  - strides 24 h (train), 24 h (validation) and 1 h (test);
  - label contract `operating_spell_labels_v3`.

  The two cache files are what training would otherwise build on its first run.

## 4. Using the outputs in the training repository

Copy these files to the same paths in `GNN_Outage_Prediction`:
- `Datasets/BPA_trm/OutagesCY2016…2023-TransmissionLineInterruptions.csv`. The outage matcher reads
  them again on every data build.
- `BPA_Line_Outage_Matcher/preprocessed_outages_2018_2023.csv`
- `model_data/bpa_graph.pkl` and `model_data/bpa_graph.json` (always both)
- `Datasets/Weather/processed/weather_processed.hdf5`, `data_mappings.json` and `scaler.pkl`
- optional: `FeatureManager/data/processed/dataset_meta_<hash>.pkl` and
  `SurvivalModels/data/processed/survival_labels_<hash>.pkl`. Training rebuilds them if they are
  missing.

**Staying comparable with study `operating_spell_v3`.** The training runner fingerprints its
inputs, and `merge` refuses runs whose inputs differ. Two outputs are not reproduced bit for bit:
- **Graph:** the pickle contains Python sets, so a rebuilt `bpa_graph.pkl` has different bytes
  even when its content is equal. That changes the `graph_pickle` fingerprint.
- **Weather:** a new download from Open-Meteo is unlikely to be byte-identical either.

To add runs to v3, reuse its `model_data/bpa_graph.pkl`/`.json` and its `Datasets/Weather/processed/`
files. A full rebuild is for a new study id.

The v3 reference values, which `labels` prints when the inputs match:

| Component | v3 value |
|---|---|
| `outage_csv` | `ec877d0d2d81e033` |
| `weather_mappings` | `8fba685bdc231d5e` |
| `weather_scaler` | `cbc2f948456f9061` |
| `weather_hdf5` | `38fa0674c069ffcc` |
| `graph_pickle` | `ac25e27dbd5abf24` |
| `model_tensors` | `760911ad2772e8f9` |
| `inputs_fingerprint` | `027c431a56ca19c7` |

## 5. Contents

| Folder | Files | Role |
|---|---|---|
| `BPA_Line_Outage_Matcher/` | `bpa_xlsx_to_trm_csv.py`, `BPA_Outage_Line_Matcher.py`, `rename_name_rules.txt` | xlsx → CSV; outage-to-line matching and Pacific→UTC |
| `BPA_Spatial_Preprocessing/` | `BPAGraphCreator.py`, `LineEndData.py`, `LineEndDictionary.py`, `compare_graphs.py`, `*_rules.txt` | line graph, its rules, and a set-order-insensitive comparison |
| `WeatherPreprocessor/` | `WeatherPreprocessor.py` | Open-Meteo download, HDF5 build, cleaning, scaling (see its `DOCUMENTATION.md`) |
| `FeatureManager/` | `OutageDataManager.py`, `outage_events.py`, `atomic_io.py`, `input_fingerprints.py` | outage matrices and event registry, graph tensors, input fingerprints |
| `SurvivalModels/` | `SurvivalOutageDataManager.py`, `contract.py` | operating-spell survival labels and the label contract |
| root | `preprocess.py` | the commands above |

`Datasets/BPA_trm/` also holds the 2014 and 2015 tables, `OutagesCY2022-TransmissionLineInterruptions_old.csv`
and `test.txt`. No step reads them.
