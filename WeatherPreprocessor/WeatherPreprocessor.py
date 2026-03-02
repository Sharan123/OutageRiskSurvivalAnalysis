import sys
import os
import requests
import pandas as pd
import numpy as np
import h5py
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from joblib import dump, load
import time
import ast
import pytz
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import partial
import traceback

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from BPA_Spatial_Preprocessing.BPAGraphCreator import BPAGraphCreator

def _clean_chunk_worker(batch_data, temporal_limit):
    """
    Worker function to clean a chunk of cells in parallel.
    Pure CPU operation: No file I/O.
    Returns:
        - batch_data: The batch data with temporal interpolation applied (modified in-place).
        - failed_indices: List of local indices (0 to batch_size-1) that still contain NaNs.
    """
    # batch_data shape: (Batch_Size, Time, Features)
    # We modify batch_data in-place to save memory (it's already a copy from IPC)
    failed_indices = []
    
    for i in range(batch_data.shape[0]):
        cell_data = batch_data[i, :, :]
        df = pd.DataFrame(cell_data)
        
        # Temporal Interpolation
        df_interpolated = df.interpolate(
            method='linear', 
            limit=temporal_limit, 
            limit_area='inside', 
            axis=0
        )
        
        # Check if NaNs remain
        if df_interpolated.isnull().values.any():
            failed_indices.append(i)
        
        batch_data[i, :, :] = df_interpolated.values.astype(np.float32)
            
    return batch_data, failed_indices

def _process_csv_batch_worker(file_paths, file_to_locations, old_id_to_new_index, features, unique_timesteps_np):
    """
    Worker to read a batch of CSV files, parse them, and prepare data for HDF5 insertion.
    Returns a tuple: (results, warnings_log)
    results: { new_sorted_idx: (time_indices, feature_data) }
    warnings_log: list of dicts with warning info
    """

    # Hardcoded dtype dict to avoid dependency on class instance
    dtype_dict = {
        'location_id': 'int32',
        'cloud_cover_high': 'int32', 'cloud_cover_low': 'int32', 'cloud_cover_mid': 'int32',
        'dew_point_2m': 'float32', 'precipitation': 'float32', 'relative_humidity_2m': 'int32',
        'rain': 'float32', 'snow_depth': 'float32', "snowfall": 'float32',
        'soil_moisture_0_to_7cm': 'float32', 'soil_temperature_0_to_7cm': 'float32',
        'surface_pressure': 'float32', 'temperature_2m': 'float32', 'weather_code': 'int32',
        'wind_direction_100m': 'int32', 'wind_direction_10m': 'int32',
        'wind_gusts_10m': 'float32', 'wind_speed_100m': 'float32', 'wind_speed_10m': 'float32',
        "vapour_pressure_deficit": 'float32', "et0_fao_evapotranspiration": 'float32',
        'time': 'object'
    }

    results = {}
    warnings_log = []

    for file_path in file_paths:
        file_path = Path(file_path)
        file_index = int(file_path.stem.split('_')[-1])
        
        if file_index not in file_to_locations:
            warnings_log.append({
                'type': 'missing_file_mapping',
                'file_index': file_index,
                'file_name': file_path.name
            })
            continue

        # Read CSV
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        try:
            empty_line_index = lines.index('\n')
        except ValueError:
            warnings_log.append({
                'type': 'malformed_file',
                'file_index': file_index,
                'file_name': file_path.name
            })
            continue # Skip malformed files

        header_raw = lines[empty_line_index + 1].strip().split(',')
        header = [h.split(' (')[0] for h in header_raw]
        
        df = pd.read_csv(
            file_path, 
            skiprows=empty_line_index + 2, 
            names=header, 
            dtype=dtype_dict
        )
        df['time'] = pd.to_datetime(df['time'])

        # Process locations in this file
        locations = file_to_locations[file_index]
        
        for local_id, original_global_id in locations:
            if original_global_id not in old_id_to_new_index:
                warnings_log.append({
                    'type': 'missing_global_id',
                    'original_global_id': original_global_id,
                    'file_index': file_index,
                    'local_id': local_id
                })
                continue
            
            new_sorted_idx = old_id_to_new_index[original_global_id]
            loc_df = df[df['location_id'] == local_id]
            
            if loc_df.empty:
                warnings_log.append({
                    'type': 'empty_location_data',
                    'local_id': local_id,
                    'file_index': file_index
                })
                continue
            
            # Extract features
            # Ensure all features exist, fill missing with NaN if necessary (though read_csv handles cols)
            # We assume features are present based on API call
            feature_data = loc_df[features].values.astype(np.float32)
            
            # Map times using searchsorted (much faster than dict lookup)
            # unique_timesteps_np must be sorted
            time_indices = np.searchsorted(unique_timesteps_np, loc_df['time'].values)
            
            results[new_sorted_idx] = (time_indices, feature_data)
            
    return results, warnings_log

def _transform_chunk_worker(raw_chunk, 
                            scaler, ohe, continuous_indices, wind_dir_indices, 
                            weathercode_idx, weathercode_ohe_features, time_features):
    """
    Worker to transform a raw data chunk.
    Pure CPU operation: No file I/O.
    """
    try:
        # raw_chunk shape: (Cells, Time, Features)
            
        # 1. Scale continuous features
        continuous_data = raw_chunk[..., continuous_indices]
        original_shape = continuous_data.shape
        continuous_data_reshaped = continuous_data.reshape(-1, len(continuous_indices))
        
        nan_mask = np.isnan(continuous_data_reshaped).any(axis=1)
        
        continuous_data_scaled = np.full_like(continuous_data_reshaped, np.nan)
        if (~nan_mask).any():
            continuous_data_scaled[~nan_mask] = scaler.transform(continuous_data_reshaped[~nan_mask])
        continuous_data_scaled = continuous_data_scaled.reshape(original_shape)

        # 2. Sin/Cos Wind Direction
        wind_data = raw_chunk[..., wind_dir_indices]
        wind_data_rad = np.deg2rad(wind_data)
        wind_sin = np.sin(wind_data_rad)
        wind_cos = np.cos(wind_data_rad)
        wind_transformed = np.stack([wind_sin[..., 0], wind_cos[..., 0], 
                                     wind_sin[..., 1], wind_cos[..., 1]], axis=-1)

        # 3. One-Hot Weathercode
        weathercode_data = raw_chunk[..., weathercode_idx].reshape(-1, 1)
        nan_mask_wc = np.isnan(weathercode_data).any(axis=1)
        
        weathercode_ohe = np.full((weathercode_data.shape[0], len(weathercode_ohe_features)), np.nan)
        if (~nan_mask_wc).any():
             weathercode_ohe[~nan_mask_wc] = ohe.transform(weathercode_data[~nan_mask_wc])
        weathercode_ohe = weathercode_ohe.reshape(raw_chunk.shape[0], raw_chunk.shape[1], -1)

        # 4. Time Features
        num_split_timesteps = time_features.shape[0]
        time_features_broadcasted = np.broadcast_to(
            time_features, 
            (raw_chunk.shape[0], num_split_timesteps, 4)
        )
        
        # 5. Concatenate (Cells, Time, Feats)
        processed_chunk_cells_first = np.concatenate([
            continuous_data_scaled,
            wind_transformed,
            weathercode_ohe,
            time_features_broadcasted
        ], axis=-1)
        
        # TRANSPOSE to (Time, Cells, Feats)
        processed_chunk_time_first = processed_chunk_cells_first.transpose(1, 0, 2)
        
        return processed_chunk_time_first
        
    except Exception as e:
        return e

class WeatherPreprocessor:
    def __init__(self, graph: BPAGraphCreator, start_date="2016-01-01", end_date="2023-12-31", 
                 verbose=False, buffer_km: float = 30.0,raw_data_dir = base_dir/"Datasets/Weather/open-metro-cells",
                 processed_data_dir = base_dir/"Datasets/Weather/processed") -> None:
        self._verbose = verbose
        self.graph = graph
        self.raw_csv_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.raw_hdf5_path = self.processed_data_dir / "weather_raw.hdf5"
        self.processed_hdf5_path = self.processed_data_dir / "weather_processed.hdf5"
        self.hdf5_mappings_path = self.processed_data_dir / "data_mappings.json"
        self.scaler_path = self.processed_data_dir / "scaler.pkl"


        self.dataset_start_date= pd.to_datetime(start_date) # YYYY-MM-DD, Inclusive
        self.dataset_end_date= pd.to_datetime(end_date) #Also inclusive

        self._global_mapping_save_file = self.raw_csv_dir / "mapping_coords_to_globalid.pkl"
        # Buffer (km) around lines for selecting 0.1° grid points. Set 0 to disable.
        self.buffer_km = buffer_km

        self.raw_csv_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.features = [
            "temperature_2m", "relative_humidity_2m", "weather_code",
            "surface_pressure", "dew_point_2m", "precipitation",
            "snowfall", "rain", "snow_depth",
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "et0_fao_evapotranspiration", "wind_speed_10m", "wind_speed_100m",
            "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
            "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm", 
            "vapour_pressure_deficit"
        ]
        self.unique_cells = []


    ############## DATA DOWNLOAD ################################
    
    def generate_api_strings(self, latitudes, longitudes, max_len=1000, api_key=None):
        features_to_download = ",".join(self.features)
        if api_key is None:
            left_part = 'https://archive-api.open-meteo.com/v1/archive?latitude='
            right_part = f"&start_date={self.dataset_start_date.strftime('%Y-%m-%d')}&end_date={self.dataset_end_date.strftime('%Y-%m-%d')}&hourly={features_to_download}&timezone=GMT&format=csv"
        else:
            left_part = 'https://customer-archive-api.open-meteo.com/v1/archive?latitude='
            right_part = f"&start_date={self.dataset_start_date.strftime('%Y-%m-%d')}&end_date={self.dataset_end_date.strftime('%Y-%m-%d')}&hourly={features_to_download}&timezone=GMT&apikey={api_key}&format=csv"
        idx = 0
        total = len(latitudes)
        while idx < len(latitudes):
            next_idx = idx
            last_valid = idx - 1
            current = None
            while next_idx < total:
                candidate = left_part + ','.join(latitudes[idx:next_idx+1]) + '&longitude=' + ','.join(longitudes[idx:next_idx+1]) + right_part
                if len(candidate) > max_len:
                    break
                current = candidate
                last_valid = next_idx
                next_idx += 1
            yield current
            idx = last_valid + 1 #next_idx

    def obtain_partially(self, api_key=None):
        cells = self.graph.get_cells_coverage(self.buffer_km)
        latitudes = [str(cell[1]) for cell in cells]
        longitudes = [str(cell[0]) for cell in cells]
        if self._verbose:
            # Print lat,lon in a zip format
            for lat, lon in zip(latitudes, longitudes):
                print(f"({lat},{lon})")

        # Pre-compute API strings to know total number of calls
        api_strings = list(self.generate_api_strings(latitudes, longitudes, 1000, api_key))
        num_api_calls = len(api_strings)
        print(f"Total API calls required: {num_api_calls}")

        # Process with progress bar showing "processing #/num_api_calls"
        with tqdm(total=num_api_calls, desc=f"processing 0/{num_api_calls}") as pbar:
            for i, api_str in enumerate(api_strings):
                pbar.set_description(f"processing {i+1}/{num_api_calls}")

                # Skip if file already exists for this index
                if self._verbose:
                    print('API String:', api_str)
                if Path(self.raw_csv_dir / f"all_cells_gmt0_{i}.csv").is_file():
                    print(f"File already exists for idx {i}. Not Saving.")
                    pbar.update(1)
                    continue

                response = requests.get(api_str, headers={'User-agent': 'Weather Batch Crawler 0.7'})

                if response.status_code == 429:  # Too many requests
                    print("Too many requests, waiting 1 second...")
                    time.sleep(1)
                    response = requests.get(api_str)
                    return False

                if response.status_code != 200:
                    print(f"Error on idx={i}. Code {response.status_code}. Not Saving...")
                else:
                    # Use pathlib for handling file paths
                    data = response.content
                    print(f"Writing for Idx:{i}")
                    output_file = Path(self.raw_csv_dir / f'all_cells_gmt0_{i}.csv')
                    output_file.write_bytes(data)

                pbar.update(1)
    
    def _read_csv_weather_part(self, file_path: Path):
        """
        Helper function to read the weather data part of a single raw CSV file.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Find the empty line index which separates the two tables
        empty_line_index = lines.index('\n')
        
        # Extract the weather data which is after the empty line
        weather_data_header_raw = lines[empty_line_index + 1].strip().split(',')
        # Clean headers (e.g., "temperature_2m (°C)" -> "temperature_2m")
        weather_data_header = [header.split(' (')[0] for header in weather_data_header_raw]

        # Use pathlib to pass the path to pandas
        dtype_dict = self.get_dtype_dict()
        
        # Ensure 'time' is read as object, not parsed immediately
        dtype_dict['time'] = 'object' 
        
        weather_df = pd.read_csv(
            file_path, 
            skiprows=empty_line_index + 2, 
            names=weather_data_header, 
            dtype=dtype_dict
        )
        return weather_df

    def create_raw_hdf5(self, force_create=False, max_workers=None, batch_size=5):
        """
        Reads all downloaded CSVs chunk by chunk and builds the
        consolidated raw HDF5 file directly. Does so directly from csv chunks.
        Should usually be performed as 2nd step after csv chunk data collection.
        """
        if self.raw_hdf5_path.exists() and self.hdf5_mappings_path.exists() and not force_create:
            print(f"Skipping, {self.raw_hdf5_path.name} and {self.hdf5_mappings_path.name} already exist.")
            return

        print("Starting memory-efficient HDF5 raw file creation (Parallel)...")

        # --- 1. Check for prerequisites ---
        print("Checking for detailed mapping file...")
        detailed_mapping = self.create_detailed_mapping()
        
        file_paths = sorted(self.raw_csv_dir.glob('all_cells_gmt0_*.csv'), key=lambda x: int(x.stem.split('_')[-1]))
        if not file_paths:
            print("No raw CSV files found. Run obtain_partially() first.")
            return
        
        # Invert mapping for easier lookup: file_index -> list of (local_id, global_id)
        file_to_locations = {}
        for (file_idx, local_id, original_global_id, _) in detailed_mapping.values():
            if file_idx not in file_to_locations:
                file_to_locations[file_idx] = []
            file_to_locations[file_idx].append((local_id, original_global_id))

        # Get canonical time index from the first file
        print("Reading time index from first CSV...")
        df_time_helper = self._read_csv_weather_part(file_paths[0])
        df_time_helper = df_time_helper[df_time_helper['location_id'] == 0] # Data for first location
        df_time_helper['time'] = pd.to_datetime(df_time_helper['time'])
        
        unique_timesteps = sorted(df_time_helper['time'].unique())
        unique_timesteps_np = np.array(unique_timesteps, dtype='datetime64[ns]') # For fast searchsorted
        timestep_to_index = {ts: i for i, ts in enumerate(unique_timesteps)}
        
        num_cells = len(detailed_mapping)
        num_timesteps = len(unique_timesteps)
        num_features = len(self.features)
        
        print(f"HDF5 shape will be: (Cells: {num_cells}, Timesteps: {num_timesteps}, Features: {num_features})")
        # --- 3. Create HDF5 mappings (CORRECTED SORTING) ---
        # detailed_mapping keys are (lat_str, lon_str).
        # We need to sort by (float(lon), float(lat)) to match OutageDataManager.
        
        # 1. Get keys and sort them correctly
        # key[0] is Lat, key[1] is Lon
        sorted_cells_keys = sorted(
            detailed_mapping.keys(), 
            key=lambda k: (float(k[1]), float(k[0])) 
        )

        # 2. Assign indices based on this sorted order
        cell_to_index = {
            str(cell_key): i 
            for i, cell_key in enumerate(sorted_cells_keys)
        }

        # 3. Create a lookup to map the 'original_global_id' (from detailed_mapping) to this new 'sorted_index'
        old_id_to_new_index = {}
        for cell_key in sorted_cells_keys:
             # detailed_mapping[cell_key] is (file_index, location_id, original_global_id, actual_loc)
             old_global_id = detailed_mapping[cell_key][2]
             old_id_to_new_index[old_global_id] = cell_to_index[str(cell_key)]

        index_to_timestep = {i: str(ts) for i, ts in enumerate(unique_timesteps)}
        
        hdf5_mappings = {
            'timestep_to_index': {str(k): v for k, v in timestep_to_index.items()},
            'index_to_timestep': index_to_timestep,
            'cell_to_index': cell_to_index, # Uses (str(lat), str(lon)) as keys
            'features_raw': self.features
        }
        with open(self.hdf5_mappings_path, 'w') as f:
            json.dump(hdf5_mappings, f, indent=2)
        print(f"Saved data mappings to {self.hdf5_mappings_path.name}")

        # --- 4. Create and populate HDF5 file ---
        print(f"Creating and populating {self.raw_hdf5_path.name}...")
        
        # Create batches of files
        file_batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
        
        with h5py.File(self.raw_hdf5_path, 'w') as hf:
            dset = hf.create_dataset(
                'weather_data', 
                shape=(num_cells, num_timesteps, num_features),
                dtype=np.float32, 
                fillvalue=np.nan,
                compression='gzip'
            )
            hf.attrs['features'] = np.array(self.features, dtype='S')

            # Parallel Processing
            all_warnings = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(
                        _process_csv_batch_worker, 
                        batch, 
                        file_to_locations, 
                        old_id_to_new_index, 
                        self.features, 
                        unique_timesteps_np
                    ): batch 
                    for batch in file_batches
                }
                
                for future in tqdm(as_completed(future_to_batch), total=len(file_batches), desc="Processing CSV Batches"):
                    try:
                        results, batch_warnings = future.result()
                        all_warnings.extend(batch_warnings)
                        
                        # Write results to HDF5 (Sequential Write, Parallel Read)
                        for new_sorted_idx, (time_indices, feature_data) in results.items():
                            dset[new_sorted_idx, time_indices, :] = feature_data
                            
                    except Exception as e:
                        print(f"Batch generated an exception: {e}")
                        raise e
            
            if all_warnings:
                print(f"\n--- Creation Warnings ({len(all_warnings)} instances) ---")
                # Group by type for cleaner output
                from collections import Counter
                type_counts = Counter(w['type'] for w in all_warnings)
                for w_type, count in type_counts.items():
                    print(f"Type '{w_type}': {count} occurrences")
                
                print("First 10 warnings:")
                for w in all_warnings[:10]:
                    print(w)

        print(f"(Step 2) Saved raw data to {self.raw_hdf5_path.name}")
    
    def clean_dataset(self, temporal_limit=24, batch_size=100, num_workers=None):
        """
        Cleans the dataset by filling missing values using:
         1. Temporal interpolation (linear) for small gaps (<temporal_limit hours) 
         2. Spatial interpolation for larger gaps - filling with average of immediate neighbors.
        Parallelized version using ProcessPoolExecutor.
        """
        print(f"Starting data cleaning (Limit: {temporal_limit}h temporal gaps)...")
        
        if not self.raw_hdf5_path.exists() or not self.hdf5_mappings_path.exists():
            print("Raw HDF5 or mappings not found. Run create_raw_hdf5() first.")
            return

        # 1. Load Mappings
        with open(self.hdf5_mappings_path, 'r') as f:
            mappings = json.load(f)
        
        # Pre-compute spatial index for fallback (Main Thread)
        index_to_cell = {}
        cell_to_index = {}
        for cell_str, idx in mappings['cell_to_index'].items():
            lat_str, lon_str = ast.literal_eval(cell_str)
            lon, lat = float(lon_str), float(lat_str)
            coord = (round(lon, 1), round(lat, 1))
            index_to_cell[idx] = coord
            cell_to_index[coord] = idx
        
        features_list = mappings.get('features_raw', [])
        index_to_timestep = mappings.get('index_to_timestep', {})

        # 2. Determine number of cells
        with h5py.File(self.raw_hdf5_path, 'r') as hf:
            num_cells = hf['weather_data'].shape[0]
            
        # 3. Create Batches
        all_indices = list(range(num_cells))
        batches = [all_indices[i:i + batch_size] for i in range(0, num_cells, batch_size)]
        
        # 4. Process in Parallel
        workers = num_workers or os.cpu_count() or 4
        print(f"Processing {num_cells} cells in {len(batches)} batches using {workers} workers.")
        max_pending_batches = workers * 2 
        
        all_warnings = []
        
        with h5py.File(self.raw_hdf5_path, 'r+') as hf:
            dataset = hf['weather_data']

            # No initializer needed for pure CPU workers
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {}
                batch_iter = iter(batches)
                
                # Initial population
                for _ in range(max_pending_batches):
                    try:
                        batch_indices = next(batch_iter)
                        start_idx = batch_indices[0]
                        end_idx = batch_indices[-1] + 1
                        
                        # Read Data (Main Thread)
                        batch_data = dataset[start_idx:end_idx, :, :]
                        
                        # Submit to Worker (CPU only)
                        fut = executor.submit(_clean_chunk_worker, batch_data, temporal_limit)
                        futures[fut] = batch_indices
                    except StopIteration:
                        break

                # Progress bar   
                with tqdm(total=len(batches), desc="Cleaning Batches") as pbar:
                    while futures:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        
                        for future in done:
                            batch_indices = futures.pop(future)
                            try:
                                processed_data, failed_local_indices = future.result()
                                
                                # Write back (Main Thread)
                                start_idx = batch_indices[0]
                                end_idx = batch_indices[-1] + 1
                                dataset[start_idx:end_idx, :, :] = processed_data
                                
                                # Handle Spatial Fallback (Main Thread)
                                for local_idx in failed_local_indices:
                                    global_idx = start_idx + local_idx
                                    current_coord = index_to_cell[global_idx]
                                    
                                    # Get neighbors
                                    neighbors_data = []
                                    offsets = [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]
                                    for d_lon, d_lat in offsets:
                                        neighbor_coord = (round(current_coord[0] + d_lon, 1), 
                                                          round(current_coord[1] + d_lat, 1))
                                        if neighbor_coord in cell_to_index:
                                            n_idx = cell_to_index[neighbor_coord]
                                            neighbors_data.append(dataset[n_idx, :, :])
                                    
                                    # Interpolate
                                    cell_df = pd.DataFrame(processed_data[local_idx])
                                    if neighbors_data:
                                        neighbors_stack = np.stack(neighbors_data, axis=0)
                                        with warnings.catch_warnings():
                                            warnings.simplefilter("ignore", category=RuntimeWarning)
                                            spatial_mean = np.nanmean(neighbors_stack, axis=0)
                                        cell_df = cell_df.fillna(pd.DataFrame(spatial_mean))
                                    
                                    # Final check and fill 0
                                    if cell_df.isnull().values.any():
                                        # Log warning
                                        nan_mask = cell_df.isnull()
                                        nan_cols = nan_mask.any(axis=0)
                                        for col_idx in nan_cols[nan_cols].index:
                                            first_nan_idx = nan_mask[col_idx].idxmax()
                                            feature_name = features_list[col_idx] if col_idx < len(features_list) else f"feat_{col_idx}"
                                            timestep_str = index_to_timestep.get(str(first_nan_idx), str(first_nan_idx))
                                            all_warnings.append({
                                                'cell_idx': global_idx,
                                                'coord': current_coord,
                                                'feature': feature_name,
                                                'timestep': timestep_str
                                            })
                                        cell_df = cell_df.fillna(0)
                                    
                                    # Write corrected cell back
                                    dataset[global_idx, :, :] = cell_df.values.astype(np.float32)

                            except Exception as e:
                                print(f"Error in batch {batch_indices}: {e}")
                                traceback.print_exc()
                            
                            pbar.update(1)
                            
                            # Submit next
                            try:
                                new_batch_indices = next(batch_iter)
                                start_idx = new_batch_indices[0]
                                end_idx = new_batch_indices[-1] + 1
                                new_batch_data = dataset[start_idx:end_idx, :, :]
                                
                                fut = executor.submit(_clean_chunk_worker, new_batch_data, temporal_limit)
                                futures[fut] = new_batch_indices
                            except StopIteration:
                                pass

        print("Dataset cleaning complete.")
        
        if all_warnings:
            print(f"\n--- Cleaning Warnings ({len(all_warnings)} instances of fallback to 0) ---")
            for w in all_warnings[:20]:
                print(f"Warning: Unable to resolve NaN for cell {w['cell_idx']} (coord: {w['coord']}), "
                      f"feature '{w['feature']}' starting at timestep {w['timestep']}. Filled with 0.")
            if len(all_warnings) > 20:
                print(f"... and {len(all_warnings) - 20} more.")
    
    def create_detailed_mapping(self, save=False):
        """
        Create a detailed mapping for each desired location to file index, local and global location IDs, and actual coordinates.
        mapping[(str(lat), str(lon))] = (file_index, location_id, global_id, (actual_lat, actual_lon)
        :param directory_path: Path to the directory containing the CSV files.
        :param desired_latitudes: List of desired latitude strings.
        :param desired_longitudes: List of desired longitude strings.
        :return: A dictionary mapping desired_location to (file_id, location_id, global_id, actual_location)
        """
        mapping = {}
        global_index = 0

        if not save and self._global_mapping_save_file.is_file():
            if self._verbose:
                print("Loading mapping from file...")
            return self.load_mapping()

        cells = self.graph.get_cells_coverage(self.buffer_km)
        desired_latitudes = [str(cell[1]) for cell in cells]
        desired_longitudes = [str(cell[0]) for cell in cells]
        # Sort files by numerical part of the filename to ensure correct order
        file_paths = sorted(self.raw_csv_dir.glob('all_cells_gmt0_*.csv'), key=lambda x: int(x.stem.split('_')[-1]))
        for file_path in tqdm(file_paths, desc="Mapping weather CSVs"):
            file_index = int(file_path.stem.split('_')[-1])

            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Find the empty line index which separates the two tables
            empty_line_index = lines.index('\n')
            
            # Separate the location data
            location_data = lines[:empty_line_index]
            
            # Create DataFrame from the location data
            location_df = pd.DataFrame([x.strip().split(',') for x in location_data[1:]],
                                    columns=location_data[0].strip().split(','))

            # Iterate over each row in the DataFrame
            for location_id, row in location_df.iterrows():
                if global_index < len(desired_latitudes):  # Check if there are still desired coordinates to map
                    actual_lat = str(row['latitude'])
                    actual_lon = str(row['longitude'])
                    actual_location = (actual_lat, actual_lon)

                    desired_location = (desired_latitudes[global_index], desired_longitudes[global_index])
                    
                    # Store in mapping
                    mapping[desired_location] = (file_index, location_id, global_index, actual_location)
                else:
                    print(f"Warning: More locations in file than desired coordinates at global index {global_index} at {file_path.name}")
                global_index += 1  # Increment global index for each location_id processed

        if save:
            self.save_mapping(mapping)
        return mapping


    def save_mapping(self, mapping):
        """
        Save the mapping dictionary to a file using pickle.

        :param mapping: The mapping dictionary to be saved.
        :param filename: The filename where the mapping will be stored.
        """
        with open(self._global_mapping_save_file, 'wb') as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Mapping saved successfully to {self._global_mapping_save_file}")

    def load_mapping(self):
        """
        Load a mapping dictionary from a pickle file.

        :param filename: The filename from which to load the mapping.
        :return: The loaded mapping dictionary.
        """
        with open(self._global_mapping_save_file, 'rb') as handle:
            mapping = pickle.load(handle)
            print(f"Mapping loaded successfully from {self._global_mapping_save_file}")
        return mapping

    @staticmethod
    def get_dtype_dict():
        return {
            'location_id': 'int32',
            'cloud_cover_high': 'int32',
            'cloud_cover_low': 'int32',
            'cloud_cover_mid': 'int32',
            'dew_point_2m': 'float32',
            'precipitation': 'float32',
            'relative_humidity_2m': 'int32',
            'rain': 'float32',
            'snow_depth': 'float32',
            "snowfall": 'float32',
            'soil_moisture_0_to_7cm': 'float32',
            'soil_temperature_0_to_7cm': 'float32',
            'surface_pressure': 'float32',
            'temperature_2m': 'float32',
            'weather_code': 'int32',
            'wind_direction_100m': 'int32',
            'wind_direction_10m': 'int32',
            'wind_gusts_10m': 'float32',
            'wind_speed_100m': 'float32',
            'wind_speed_10m': 'float32',
            "vapour_pressure_deficit": 'float32',
            "et0_fao_evapotranspiration": 'float32'
        }
    
    def _get_time_features(self, timesteps_str):
        """Helper to generate sin/cos time features from a list of datetime strings."""
        timesteps = pd.to_datetime(timesteps_str)
        
        # Day of Year
        day_of_year = timesteps.dayofyear
        days_in_year = np.where(timesteps.is_leap_year, 366.0, 365.0)
        day_of_year_sin = np.sin(2 * np.pi * day_of_year / days_in_year)
        day_of_year_cos = np.cos(2 * np.pi * day_of_year / days_in_year)

        # Hour of Week
        hour_of_week = timesteps.dayofweek * 24 + timesteps.hour
        hour_of_week_sin = np.sin(2 * np.pi * hour_of_week / (7 * 24))
        hour_of_week_cos = np.cos(2 * np.pi * hour_of_week / (7 * 24))
        
        return np.stack([
            day_of_year_sin, day_of_year_cos, 
            hour_of_week_sin, hour_of_week_cos
        ], axis=-1).astype(np.float32)
    
    def _warn_nan_entries(self, feature_group, nan_row_indices, cell_chunk_slice, timesteps_for_split, max_examples=10):
        """
        Emit a warning when NaNs are encountered during preprocessing.
        Reports the global cell index and timestep for a subset of occurrences.
        """
        if nan_row_indices.size == 0:
            return
        
        num_timesteps = len(timesteps_for_split)
        cell_offsets, time_offsets = np.divmod(nan_row_indices, num_timesteps)
        chunk_start = cell_chunk_slice.start or 0
        
        details = []
        for idx in range(min(max_examples, nan_row_indices.size)):
            global_cell_idx = chunk_start + int(cell_offsets[idx])
            timestep_label = timesteps_for_split[int(time_offsets[idx])]
            details.append(f"(cell {global_cell_idx}, timestep {timestep_label})")
        
        detail_str = ", ".join(details)
        if nan_row_indices.size > max_examples:
            detail_str += f", ... {nan_row_indices.size - max_examples} more occurrences"
        
        print(f"Warning: NaNs detected in {feature_group} at {detail_str}")

    def _get_split_indices(self, num_timesteps, train_split, valid_split):
        """Helper to calculate train/valid/test time indices."""
        if isinstance(train_split, str):
            # Split by date
            with open(self.hdf5_mappings_path, 'r') as f:
                mappings = json.load(f)
            index_to_timestep = mappings['index_to_timestep'] # JSON only supports string keys
            timesteps = [index_to_timestep[str(i)] for i in range(num_timesteps)] # So we convert int keys
            
            train_end_idx = next(i for i, ts in enumerate(timesteps) if ts >= train_split)
            valid_end_idx = next(i for i, ts in enumerate(timesteps) if ts >= valid_split)
        
        elif isinstance(train_split, float):
            # Split by percentage
            train_end_idx = int(num_timesteps * train_split)
            valid_end_idx = int(num_timesteps * (train_split + valid_split))
        
        else:
            raise ValueError("train_split must be a date string or a float percentage")
            
        train_indices = slice(0, train_end_idx)
        valid_indices = slice(train_end_idx, valid_end_idx)
        test_indices = slice(valid_end_idx, num_timesteps)
        
        return train_indices, valid_indices, test_indices

    def transform_ds(self, 
                     scaler_type='minmax', 
                     train_split='2022-01-01T00:00', 
                     valid_split='2023-01-01T00:00',
                     chunk_size=1000,
                     num_workers=None):
        """
        Step 3 & 4: Creates a new, processed HDF5 file with transformed,
        scaled, and feature-engineered data, split into train/valid/test.
        SHAPE: (Time, Cells, Features)
        """
        print("Step 3 & 4: Starting data transformation and splitting...")
        
        if not self.raw_hdf5_path.exists() or not self.hdf5_mappings_path.exists():
            print("Raw data not found. Run create_raw_hdf5() first.")
            return

        # Load mappings
        with open(self.hdf5_mappings_path, 'r') as f:
            mappings = json.load(f)
        features_raw = mappings['features_raw']
        index_to_timestep = mappings['index_to_timestep']
        
        # Load raw data
        with h5py.File(self.raw_hdf5_path, 'r') as hf_raw:
            raw_data_shape = hf_raw['weather_data'].shape
            num_cells, num_timesteps, num_raw_features = raw_data_shape

            # --- 1. Define Split Indices ---
            splits = {}
            splits['train'], splits['valid'], splits['test'] = self._get_split_indices(
                num_timesteps, train_split, valid_split
            )
            
            if self._verbose:
                print(f"Train split: {splits['train']}, Valid split: {splits['valid']}, Test split: {splits['test']}")
            print(f"Train/Valid/Test temporal size: {splits['train'].stop}, {splits['valid'].stop - splits['valid'].start}, {splits['test'].stop - splits['test'].start}")

            # --- 2. Fit Scalers ---
            print("Fitting scalers on training data...")
            
            # Continuous features to scale
            continuous_features = [
                "temperature_2m", "relative_humidity_2m",
                "surface_pressure", "dew_point_2m", "precipitation", "snowfall", "rain",
                "snow_depth", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", 
                "et0_fao_evapotranspiration", "wind_speed_10m", "wind_speed_100m",
                "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
                "vapour_pressure_deficit"]
            continuous_indices = [features_raw.index(f) for f in continuous_features]
            
            # Wind direction features for sin/cos
            wind_dir_features = ['wind_direction_10m', 'wind_direction_100m']
            wind_dir_indices = [features_raw.index(f) for f in wind_dir_features]
            
            # Weathercode for one-hot
            weathercode_idx = features_raw.index('weather_code')

            # Initialize scalers
            if scaler_type.lower() == 'standard':
                scaler = StandardScaler()
            elif scaler_type.lower() == 'minmax':
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                raise ValueError("scaler_type must be 'standard' or 'minmax'")
                
            # Fit scaler in chunks (to save memory)
            train_slice = splits['train']
            for i in range(0, num_cells, chunk_size):
                chunk = hf_raw['weather_data'][i:i+chunk_size, train_slice, :]
                chunk_continuous = chunk[..., continuous_indices]
                # Reshape to 2D (samples, features) and filter NaNs
                chunk_reshaped = chunk_continuous.reshape(-1, len(continuous_indices)) # (samples, features) where samples = chunk_size (cells) * timesteps
                chunk_reshaped = chunk_reshaped[~np.isnan(chunk_reshaped).any(axis=1)] # filter out rows with NaNs
                if chunk_reshaped.shape[0] > 0:
                    scaler.partial_fit(chunk_reshaped)
            
            # Fit OneHotEncoder for weathercode
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            all_weathercodes = []
            for i in range(0, num_cells, chunk_size):
                chunk = hf_raw['weather_data'][i:i+chunk_size, train_slice, weathercode_idx]
                chunk_reshaped = chunk.reshape(-1, 1)
                all_weathercodes.append(chunk_reshaped)
            all_weathercodes_np = np.concatenate(all_weathercodes, axis=0)
            all_weathercodes_np = all_weathercodes_np[~np.isnan(all_weathercodes_np).any(axis=1)]
            ohe.fit(all_weathercodes_np)
            weathercode_ohe_features = ohe.get_feature_names_out(['weather_code'])
            
            # Save the fitted scaler
            dump(scaler, self.scaler_path)
            print(f"Saved fitted scaler to {self.scaler_path}")
            # Note: OHE is saved implicitly via the transformed data shape

            # --- 3. Define Final Processed Features ---
            final_features = continuous_features + \
                             ['wind_dir_10m_sin', 'wind_dir_10m_cos', 'wind_dir_100m_sin', 'wind_dir_100m_cos'] + \
                             list(weathercode_ohe_features) + \
                             ['day_of_year_sin', 'day_of_year_cos', 'hour_of_week_sin', 'hour_of_week_cos']
            
            
            # --- 4. Create Processed HDF5 ---
            print(f"Creating processed HDF5 file at {self.processed_hdf5_path.name}...")
            
            workers = num_workers or os.cpu_count() or 4
            # Adjust chunk size if parallel to avoid huge IPC payloads
            # If user didn't specify a small chunk size, we force it smaller for parallel exec
            if workers > 1 and chunk_size > 20:
                print(f"Adjusting chunk_size from {chunk_size} to 20 for parallel processing efficiency.")
                chunk_size = 20

            with h5py.File(self.processed_hdf5_path, 'w') as hf_proc:
                
                hf_proc.attrs['features'] = np.array(final_features, dtype='S')
                
                # Process and save each split
                for split_name, time_slice in splits.items():
                    print(f"Processing split: {split_name}")
                    
                    # Get time features for this split
                    split_timesteps_idx = list(range(num_timesteps))[time_slice] # get actual indices
                    split_timesteps_str = [index_to_timestep[str(i)] for i in split_timesteps_idx]
                    time_features = self._get_time_features(split_timesteps_str) # Shape: (T_split, 4)
                    
                    # Get split shape
                    num_split_timesteps = len(split_timesteps_str)

                    # SHAPE: (Time, Cells, Features)
                    split_shape = (num_split_timesteps, num_cells, len(final_features))
                    # CHUNK: (24 hours, All_Cells, All_Features)
                    chunk_shape = (min(24, num_split_timesteps), num_cells, len(final_features))
                    
                    # Create dataset for this split
                    dset = hf_proc.create_dataset(
                        split_name, 
                        shape=split_shape, 
                        dtype=np.float32, 
                        chunks=chunk_shape,
                        compression='gzip'
                    )
                    
                    # Create batches of cell indices
                    cell_batches = [(i, min(i + chunk_size, num_cells)) for i in range(0, num_cells, chunk_size)]
                    
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        futures = {}
                        # Submit initial batch
                        for i, (start, end) in enumerate(cell_batches):
                            if i >= workers * 2: break
                            
                            # Read raw chunk (Main Thread)
                            raw_chunk = hf_raw['weather_data'][start:end, time_slice, :]
                            
                            fut = executor.submit(
                                _transform_chunk_worker,
                                raw_chunk,
                                scaler, ohe, continuous_indices, wind_dir_indices,
                                weathercode_idx, weathercode_ohe_features, time_features
                            )
                            futures[fut] = (start, end)
                            
                        next_batch_idx = workers * 2
                        
                        with tqdm(total=len(cell_batches), desc=f"Writing {split_name} split") as pbar:
                            while futures:
                                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                                
                                for future in done:
                                    start, end = futures.pop(future)
                                    result = future.result()
                                    
                                    if isinstance(result, Exception):
                                        print(f"Error in batch {start}-{end}: {result}")
                                        raise result
                                    
                                    # Write to dataset (Time, Cells, Feats)
                                    # result is (Time, Cells_in_chunk, Feats)
                                    dset[:, start:end, :] = result
                                    pbar.update(1)
                                    
                                    # Submit next
                                    if next_batch_idx < len(cell_batches):
                                        n_start, n_end = cell_batches[next_batch_idx]
                                        
                                        # Read raw chunk (Main Thread)
                                        raw_chunk = hf_raw['weather_data'][n_start:n_end, time_slice, :]
                                        
                                        fut = executor.submit(
                                            _transform_chunk_worker,
                                            raw_chunk,
                                            scaler, ohe, continuous_indices, wind_dir_indices,
                                            weathercode_idx, weathercode_ohe_features, time_features
                                        )
                                        futures[fut] = (n_start, n_end)
                                        next_batch_idx += 1

        print("Step 3 & 4: Data transformation complete.")
        print(f"Processed file saved to: {self.processed_hdf5_path.name}")
        print(f"Final features: {final_features}")
    
    @staticmethod
    def convert_time(time_str): 
        """ Convert time from GMT to Pacific Time considering DST """
        gmt = pytz.timezone('GMT')
        pacific = pytz.timezone('America/Los_Angeles')
        # Parse the time string into a datetime object in GMT
        datetime_gmt = gmt.localize(pd.to_datetime(time_str))
        # Convert to Pacific time
        datetime_pacific = datetime_gmt.astimezone(pacific)
        return datetime_pacific.strftime('%Y-%m-%dT%H:%M')
    ############## END OF DATA DOWNLOAD #########################

    def load_scaller(self):
        """
        Load the scaler from the standard processed path.
        """
        if self.scaler_path.is_file():
            print(f"Loading scaler from {self.scaler_path}")
            return load(self.scaler_path)
        else:
            raise FileNotFoundError(f"No scaler found at {self.scaler_path}. Run transform_ds() first.")
    
    def _load_mappings_if_needed(self):
        """Helper to load and cache mappings."""
        if not hasattr(self, '_cached_mappings'):
            if not self.hdf5_mappings_path.exists():
                raise FileNotFoundError(f"Mappings file not found at {self.hdf5_mappings_path}")
            
            with open(self.hdf5_mappings_path, 'r') as f:
                self._cached_mappings = json.load(f)
            
            # Pre-calculate split boundaries based on HDF5 file structure
            # This allows us to map Global Time Index -> Local Split Index
            if self.processed_hdf5_path.exists():
                with h5py.File(self.processed_hdf5_path, 'r') as hf:
                    self._split_bounds = {}
                    current_offset = 0
                    # Order matters: train -> valid -> test
                    for split in ['train', 'valid', 'test']:
                        if split in hf:
                            length = hf[split].shape[0] # Shape is (Time, Cells, Features)
                            self._split_bounds[split] = (current_offset, current_offset + length)
                            current_offset += length
                        else:
                            print(f"Warning: Split '{split}' not found in processed HDF5 file.")
            else:
                raise FileNotFoundError(f"Processed HDF5 not found at {self.processed_hdf5_path}")

    def get_split_boundaries(self):
        """
        Returns the start and end indices for each split in the global timeline.
        Returns:
            dict: {'train': (start, end), 'valid': (start, end), 'test': (start, end)}
        """
        self._load_mappings_if_needed()
        return self._split_bounds

    def fetch_vectorized(self, global_time_idx, window_size):
        """
        Fetches a time window of weather data for ALL cells.
        Optimized for the (Time, Cells, Features) HDF5 structure.
        
        Args:
            global_time_idx (int): The starting index in the GLOBAL timeline (0 to Total_Steps).
            window_size (int): Number of timesteps to fetch.
        
        Returns:
            np.ndarray: Shape (Window_Size, Num_Cells, Num_Features)
        """
        self._load_mappings_if_needed()
        
        # Determine which split this global index belongs to
        target_split = None
        local_start = 0
        
        for split, (bound_start, bound_end) in self._split_bounds.items():
            if bound_start <= global_time_idx < bound_end:
                # Check if window fits entirely within this split
                if global_time_idx + window_size <= bound_end:
                    target_split = split
                    local_start = global_time_idx - bound_start
                    break
                else:
                    raise ValueError(f"Requested window crosses split boundaries or exceeds dataset.")
        
        if target_split is None:
            raise ValueError(f"Time index {global_time_idx} out of bounds.")
            
        with h5py.File(self.processed_hdf5_path, 'r') as hf:
            # Efficient slice
            data_slice = hf[target_split][local_start : local_start + window_size, :, :]
            
        return data_slice
# # WMO CODE descriptions
# Code 	Description
# 0 	Clear sky
# 1, 2, 3 	Mainly clear, partly cloudy, and overcast
# 45, 48 	Fog and depositing rime fog
# 51, 53, 55 	Drizzle: Light, moderate, and dense intensity
# 56, 57 	Freezing Drizzle: Light and dense intensity
# 61, 63, 65 	Rain: Slight, moderate and heavy intensity
# 66, 67 	Freezing Rain: Light and heavy intensity
# 71, 73, 75 	Snow fall: Slight, moderate, and heavy intensity
# 77 	Snow grains
# 80, 81, 82 	Rain showers: Slight, moderate, and violent
# 85, 86 	Snow showers slight and heavy
# 95 * 	Thunderstorm: Slight or moderate
# 96, 99 * 	Thunderstorm with slight and heavy hail


# Code 	Description
# 0 	Clear
# 1 	Mostly Clear
# 2 	Partly Cloudy
# 3 	Cloudy
# 45 	Fog
# 48 	Freezing Fog
# 51 	Light Drizzle
# 53 	Drizzle
# 55 	Heavy Drizzle
# 56 	Light Freezing Drizzle
# 57 	Freezing Drizzle
# 61 	Light Rain
# 63 	Rain
# 65 	Heavy Rain
# 66 	Light Freezing Rain
# 67 	Freezing Rain
# 71 	Light Snow
# 73 	Snow
# 75 	Heavy Snow
# 77 	Snow Grains
# 80 	Light Rain Shower
# 81 	Rain Shower
# 82 	Heavy Rain Shower
# 85 	Snow Shower
# 86 	Heavy Snow Shower
# 95 	Thunderstorm
# 96 	Hailstorm
# 99 	Heavy Hailstorm
