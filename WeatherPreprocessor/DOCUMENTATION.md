# Weather Preprocessing Code Documentation

---

This document describes the `WeatherPreprocessor` class, which prepares historical weather data from Open‑Meteo for use with the BPA transmission network graph. It can:
- Determine all 0.1° grid cells intersected by BPA lines
- Download hourly weather data for those cells (free or customer API)
- Consolidate downloaded CSVs into a memory-efficient HDF5 format
- Perform temporal and spatial imputation to clean missing data
- Transform features (Scaling, One-Hot Encoding, Cyclical Time Features)
- Split data into Train/Validation/Test sets and save to a processed HDF5 file
- Provide optimized vectorized data fetching for GNN training

It integrates with the spatial graph produced by `BPAGraphCreator` and expects the BPA transmission lines at `Datasets/TransmissionLineData/BPA_TransmissionLines.geojson`.

---

## Quick Start

Example end‑to‑end usage:
```python
from BPA_Spatial_Preprocessing.BPAGraphCreator import BPAGraphCreator
from WeatherPreprocessor.WeatherPreprocessor import WeatherPreprocessor

# 1) Build/obtain the BPA graph
bpa_graph = BPAGraphCreator()  # use project defaults inside BPAGraphCreator

# 2) Initialize preprocessor for desired time window (inclusive)
#    Set buffer_km to include a corridor around lines (default 30 km)
wp = WeatherPreprocessor(bpa_graph, start_date="2016-01-01", end_date="2023-12-31", verbose=True, buffer_km=30.0)

# 3) Download data in batches (creates all_cells_gmt0_{i}.csv under Datasets/Weather/open-metro-cells)
#    If you have an Open‑Meteo customer API key, pass it to use the customer endpoint.
wp.obtain_partially(api_key=None)

# 4) Create Raw HDF5 from downloaded CSVs
#    Consolidates scattered CSVs into a single 'weather_raw.hdf5' file
wp.create_raw_hdf5()

# 5) Clean and Impute Data
#    Fills missing values using temporal interpolation (limit 2h) and spatial averaging
wp.clean_dataset(temporal_limit=2)

# 6) Transform, Scale, and Split Data
#    Fits scalers on training data, encodes categorical features, and saves 'weather_processed.hdf5'
#    Data is split into Train/Valid/Test groups within the HDF5 file
wp.transform_ds(scaler_type='minmax', train_split='2022-01-01', valid_split='2023-01-01')

# 7) Fetch vectorized features during training
#    Returns a tensor of shape (Window_Size, Num_Cells, Num_Features) for all cells
data_window = wp.fetch_vectorized(global_time_idx=0, window_size=24)
```

---

## `WeatherPreprocessor` Class

<details>
<summary>Key Methods</summary>

- <details>
  <summary><strong>`__init__(self, graph, start_date="2016-01-01", end_date="2023-12-31", verbose=False)`</strong></summary>

  Constructor that wires the BPA graph, sets the dataset time window, and initializes paths.
  - `raw_csv_dir`: `Datasets/Weather/open-metro-cells/`
  - `processed_data_dir`: `Datasets/Weather/processed/`
  - `raw_hdf5_path`: `weather_raw.hdf5`
  - `processed_hdf5_path`: `weather_processed.hdf5`
  </details>

- <details>
  <summary><strong>`obtain_partially(self, api_key=None)`</strong></summary>

  Downloads hourly weather data for all grid cells intersected by lines present in the graph. It batches requests by concatenating latitude/longitude lists to stay under URL length limits. Timezone is GMT+0. Files are saved as `all_cells_gmt0_{i}.csv`. Handles HTTP 429 by retrying.
  </details>

- <details>
  <summary><strong>`create_raw_hdf5(self, force_create=False)`</strong></summary>

  Reads all downloaded CSV chunks and builds the consolidated `weather_raw.hdf5` file.
  - Creates `data_mappings.json` which maps grid cells (lat, lon) to HDF5 indices.
  - Sorts cells by (Longitude, Latitude) to ensure consistent indexing.
  - Shape: `(Num_Cells, Num_Timesteps, Num_Features)`.
  </details>

- <details>
  <summary><strong>`clean_dataset(self, temporal_limit=2)`</strong></summary>

  Performs in-place imputation on `weather_raw.hdf5`.
  1. **Temporal Interpolation**: Linearly interpolates small gaps (<= `temporal_limit` hours).
  2. **Spatial Interpolation**: Fills remaining gaps (large outages or missing sensors) by averaging valid data from immediate spatial neighbors (up/down/left/right).
  </details>

- <details>
  <summary><strong>`transform_ds(self, scaler_type='minmax', train_split=..., valid_split=...)`</strong></summary>

  The main processing pipeline. Creates `weather_processed.hdf5`.
  - **Splitting**: Defines Train/Valid/Test ranges based on provided dates.
  - **Scaling**: Fits `StandardScaler` or `MinMaxScaler` on the **Training** split only, then applies to all.
  - **Encoding**: One-hot encodes `weather_code`.
  - **Feature Engineering**: Converts Wind Direction to Sin/Cos components; Adds Cyclical Time features (Day of Year, Hour of Week).
  - **Storage**: Saves data in chunks of `(Time, All_Cells, All_Features)` optimized for whole-grid querying.
  </details>

- <details>
  <summary><strong>`fetch_vectorized(self, global_time_idx, window_size)`</strong></summary>

  Optimized data fetcher for GNN training.
  - **Input**: Global time index and window size.
  - **Output**: `np.ndarray` of shape `(Window_Size, Num_Cells, Num_Features)`.
  - Efficiently reads contiguous chunks from HDF5 corresponding to the requested time window for *all* cells.
  </details>

</details>

<details>
<summary>Data Download & Preparation</summary>

- `_get_cells(self)`: Scans lines to generate unique 0.1° cells using `buffer_km`.
- `generate_api_strings(...)`: Yields batched API URLs.
- `create_detailed_mapping(...)`: Maps desired coordinates to file locations. Used internally by `create_raw_hdf5`.

</details>

<details>
<summary>Feature Engineering Details</summary>

The `transform_ds` method generates the following feature set in `weather_processed.hdf5`:
1. **Continuous (Scaled)**: Temperature, Humidity, Pressure, Precip, Snow, Cloud Cover, Wind Speed, Gusts, Soil Moisture/Temp, VPD, ET0.
2. **Wind Direction**: Transformed into `sin` and `cos` components (10m and 100m).
3. **Weather Code**: One-hot encoded (sparse).
4. **Time Features**: Cyclical encoding (Sin/Cos) for Day of Year and Hour of Week.

</details>

---

## Notes & Conventions

- **File Structure**:
  - `Datasets/Weather/open-metro-cells/`: Raw CSV batches.
  - `Datasets/Weather/processed/`:
    - `weather_raw.hdf5`: Consolidated raw data (with NaNs).
    - `weather_processed.hdf5`: Final clean, scaled, and engineered data.
    - `data_mappings.json`: Metadata mapping cells and times to indices.
    - `scaler.pkl`: Saved scaler object.

- **Timezones**:
  - Raw data is GMT+0.
  - `transform_ds` preserves the input timeline.
  - `OutageDataManager` handles alignment with outage data.

- **HDF5 Optimization**:
  - `weather_processed.hdf5` is chunked as `(24, Num_Cells, Num_Features)`.
  - This layout is optimized for reading *all cells* for a specific *time window*, which is the access pattern for Graph Neural Networks.
