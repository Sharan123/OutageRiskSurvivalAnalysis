import sys
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from pathlib import Path
import pickle
import numpy as np
from shapely.prepared import prep
from collections import OrderedDict, defaultdict
import re
import json

base_dir = Path(__file__).resolve().parent.parent
if base_dir not in sys.path:
    sys.path.append(str(base_dir))

from BPA_Spatial_Preprocessing.LineEndDictionary import LineEndDictionary
from BPA_Spatial_Preprocessing.LineEndData import ConnectType, LineEnd, DEBUG

class BPAGraphCreator:
    """
        Creates a graph from the BPA Transmission Line GeoJSON file containing the lines coordinates
    """
    def __init__(self, bpa_geojson_path = Path(__file__).parent.parent / "Datasets/TransmissionLineData/BPA_TransmissionLines.geojson"):
        self.lines = gpd.read_file(bpa_geojson_path)
        self._apply_renames()
        self.line_ends_dict = LineEndDictionary(self.lines)
        print('Total number of free points before filling: ', self.line_ends_dict.total_num_free_points())
        self.line_ends_dict.fill_touch_connections()
        print('Total number of free points after self-touch: ', self.line_ends_dict.total_num_free_points())
        self.line_ends_dict.fill_self_intersection_connections()
        print('Total number of free points after self-intersect: ', self.line_ends_dict.total_num_free_points())
        self.line_ends_dict._apply_hotfixes_for_substations()
        self._verbose = False

        self.cell_coverage_cache = None
        self.graph_info = None

    def set_debug(self, debug=False):
        global DEBUG
        DEBUG = debug
        self.line_ends_dict.set_debug(debug)
   
    def save(self, filepath):
        """
            Persist the current graph state, including the line geometries and end dictionary.
        """
        path = Path(filepath)
        parent = path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'lines': self.lines,
            'line_ends_dict': self.line_ends_dict.to_serializable(include_lines=False),
        }
        with path.open('wb') as file:
            pickle.dump(payload, file)

        self.get_cells_coverage()
        graph_info = self._build_graph_info()
        json_path = path.with_suffix('.json')
        with json_path.open('w') as json_file:
            json.dump(graph_info, json_file, indent=2)
        
        if self._verbose:
            print(f"Saved base graph structure to {path}")
            print(f"Saved detailed graph info to {json_path}")

    @classmethod
    def load(cls, filepath):
        """
            Restore a BPAGraphCreator from a previously saved state.
        """
        path = Path(filepath)
        with path.open('rb') as file:
            payload = pickle.load(file)

        instance = cls.__new__(cls)
        instance.lines = payload['lines']
        instance.line_ends_dict = LineEndDictionary.from_serialized(
            payload['line_ends_dict'],
            lines=instance.lines,
        )
        instance._verbose = False 
        instance.cell_coverage_cache = None
        instance.graph_info = None

        json_path = path.with_suffix('.json')
        if json_path.exists():
            if instance._verbose:
                print(f"Loading graph info from {json_path}")
            with json_path.open('r') as json_file:
                instance.graph_info = json.load(json_file)
            
            # 3. Restore the cell coverage cache from the graph info
            instance.cell_coverage_cache = OrderedDict()
            for xrefcd, data in instance.graph_info.items():
                if 'cell_coverage' in data:
                    # JSON saves tuples as lists, convert them back to tuples
                    instance.cell_coverage_cache[xrefcd] = [tuple(cell) for cell in data['cell_coverage']]
        else:
            print(f"Warning: Graph info file not found at {json_path}. Caches will be empty.")
        
        return instance


    def _apply_renames(self):
       
        # Load rename rules from file
        with open(Path(__file__).parent / 'rename_rules.txt', 'r') as f:
            rename_rules = f.readlines()

        # Process each rename rule
        for rule in rename_rules:
            # Ignore comments
            rule = rule.split(';')[0].strip()
            if not rule:
                continue  # Skip empty or comment-only lines

            original_name, new_name = rule.split('->')
            original_name = original_name.strip()
            new_name = new_name.strip()

            if new_name == 'X' or new_name == 'x':
                # Delete the line from self.lines
                self.lines = self.lines[self.lines['XRefCd'] != original_name]
            else:
                # Update the XRefCd of the line in self.lines
                self.lines.loc[self.lines['XRefCd'] == original_name, 'XRefCd'] = new_name

    # --- NEW HELPER METHODS FOR LINE TYPE (from BPA_Outage_Line_Matcher.py) ---
    @staticmethod
    def is_a_tap(name_outage):
        # Check if name_outage has a ' TAP TO ' in it and obtain left and right side of the string
        match = re.match(r'.* TAP TO .*', name_outage)
        if not match or name_outage.count('TAP TO')!=1 or name_outage.count('SECTION OF')!=0 or name_outage.endswith(' TAP'):
            return False
        return True
    
    @staticmethod
    def is_a_line(name_outage):
        if 'SECTION OF' in name_outage or 'TAP TO' in name_outage or name_outage.endswith(' TAP'):
            return False
        return True

    @staticmethod
    def is_a_tap_to_tap(name_outage):
        if name_outage.count('SECTION OF')!=0:
            return False
        elif ' TAP TO ' in name_outage and name_outage.endswith(' TAP'):
            return True
        
        return False

    def _get_line_type(self, line_name):
        """Helper to determine line type ('L', 'T', 'TT') from its name."""
        if not isinstance(line_name, str):
            return 'UNKNOWN'
        
        if BPAGraphCreator.is_a_line(line_name):
            return 'L'
        if BPAGraphCreator.is_a_tap(line_name):
            return 'T'
        if BPAGraphCreator.is_a_tap_to_tap(line_name):
            return 'TT'
        
        # Add other types like 'S' or 'TS' if needed
        print(f"Warning: Line name '{line_name}' does not match any known type. Marking as 'UNKNOWN'.")
        return 'UNKNOWN'
    # --- END NEW HELPER METHODS ---


    @staticmethod  
    def xrefCdIntersectCheck(xrefcd, merging_xrefcd):
        """
            Checks if the xrefcd and merging_xrefcd intersect according to the rules. The xrefcd is the line that merges into the merging_xrefcd
            The rules are: i) The two letters combined from merging_xrefcd need to form a 'station name' for xrefcd. 
                          ii) The station name in xrefcd has the letter 'T' appended to the first station name, at the end. Making it a 5 letter code
            Example: xrefcd = 'ABERT-SACO-1', merging_xrefcd = 'SATP-COSM-1' - it is good since SA+CO = SACO which is one substation in xrefcd; and ABERT has 5 letters and last letter is T
            Returns: True if they intersect, False otherwise
        """
        substations_xfrecd = xrefcd.split('-')[:2]
        substations_merging_xfrecd = merging_xrefcd.split('-')[:2]
        merging_xfrecd_station_str = substations_merging_xfrecd[0][:2] + substations_merging_xfrecd[1][:2]
        if len(substations_xfrecd[0]) != 5 or substations_xfrecd[0][-1] != 'T':
            return False
        
        if merging_xfrecd_station_str == substations_xfrecd[1]:
            return True
        if len(substations_xfrecd[1]) == 5 and substations_xfrecd[1][-1] == 'T' and \
           len(substations_merging_xfrecd[0])==5 and substations_merging_xfrecd[0][-1] == 'T':
            # We need to check if any substation in substations_xrefcd matches any in substations_merging_xfrecd
            for subst in substations_xfrecd:
                if subst in substations_merging_xfrecd:
                    return True
        return False
    
    def getLineStringIntersects(self):
        """
            Goes through each line in self.line_ends_dict and checks if it intersects with another line. If it does, it assigns the connection type to LINE 
            instead of free.
            Returns: None
        """
        for xrefcd, linestr_list in self.line_ends_dict.items():
            for line_str_target in linestr_list:
                for line_str_target_end in line_str_target:
                    line_str_are_equal = False
                    if line_str_target_end.connection_type != ConnectType.FREE:
                        continue
                    for _, line_search in self.lines.iterrows():
                        if xrefcd==line_search['XRefCd']:
                            continue
                        if not line_search.geometry.intersects(line_str_target_end.associated_line_str):
                            continue
                        if not BPAGraphCreator.xrefCdIntersectCheck(xrefcd, line_search['XRefCd']):
                            continue
                        line_string = self.lines[self.lines['XRefCd'] == xrefcd].iloc[0].geometry.geoms[line_str_target_end.line_idx]
                        if line_string.equals(line_str_target_end.associated_line_str):
                            intersection_point = line_search.geometry.intersection(line_str_target_end.associated_line_str)
                            line_str_are_equal = True
                            if Point(line_string.coords[0]).distance(intersection_point) < Point(line_string.coords[-1]).distance(intersection_point):
                                line_str_target_end = line_str_target[0]
                            else:
                                line_str_target_end = line_str_target[1]
                        line_str_target_end.connection_type = ConnectType.LINE
                        line_str_target_end.all_merging_lines_xrefcd.add(line_search['XRefCd'])
                        line_str_target_end.free_point = False
                        self.line_ends_dict.remove_substation(xrefcd, line_search['XRefCd'])
                        if DEBUG:
                            print(f'LineString {xrefcd} intersects with {line_search["XRefCd"]}')
                            print(f'{xrefcd}, {line_str_target_end}')
                    if line_str_are_equal:
                        break
    # Method which goes through each line in self.line_ends_dict and checks if there is one unasigned part of the XRefCd, we assign it to the substation
    def assignSubstationToRemainingFreePoint(self):
        """
            Goes through each line in self.line_ends_dict and checks if there is one unasigned part of the XRefCd and one non-matched substation, 
            if so we assign the unassigned point to the substation
            Returns: None
        """
        for xrefcd, linestr_list in self.line_ends_dict.items():
            # if number of free points is not 1 continue
            if self.line_ends_dict.get_num_free_points(xrefcd) != 1 or len(self.line_ends_dict.line_subs_dict[xrefcd]) != 1:
                continue
            for line_str_target in linestr_list:
                for line_str_target_pt in line_str_target:
                    if line_str_target_pt.free_point:
                        line_str_target_pt.connection_type = ConnectType.SUBSTATION
                        line_str_target_pt.free_point = False
                        # Check which part of xrefcd is not assigned 
                        station_name = self.line_ends_dict.line_subs_dict[xrefcd].pop()
                        line_str_target_pt.substation_name = station_name
                        if DEBUG:
                            print(f'LineString {xrefcd} is assigned to a substation {station_name}')
                        break
    
    @staticmethod
    def is_name_matching(substation_name, line_xrefcd):
            """
                Checks if the substation name matches the XRefCd. The substation name is formed by merging the first two letters of the XRefCd
                Returns: True if the substation name matches the XRefCd, False otherwise
            """
            parts = line_xrefcd.split('-')
            # Check if either of the two parts is 5 letters with the 5th letter being 'T'; If so crop the last letter
            if len(parts[0]) == 5 and parts[0][-1] == 'T':
                parts[0] = parts[0][:-1]
            if len(parts[1]) == 5 and parts[1][-1] == 'T':
                parts[1] = parts[1][:-1]
            # Check if either of the first two parts of the XRefCd matches the substation name
            return substation_name in parts[:2]
        
    def match_proximity_substations(self, proximity=200):
        """
            Matches substations to line ends based on proximity and name matching. 
            If a line end is within the given proximity and has a matching substation name, it is assigned to that substation. Only ONE line end from each line can be assigned to a substation
            Modifies: self.line_ends_dict
            Returns: None
        """
        
        changes_made = True
        cnt_matches = 0
        while changes_made:
            changes_made = False
            assignment_tracker = {}  # Tracks line assignments to prevent double assignments within iterations

            for xrefcd, line_ends in self.line_ends_dict.items():
                for line_end_list in line_ends:
                    for line_end_data in line_end_list:
                        # Skip free points and line ends without a matched substation
                        if line_end_data.free_point or line_end_data.substation_name is None:
                            continue
                        
                        # Search for nearby free line ends
                        for other_xrefcd, other_line_ends in self.line_ends_dict.items():
                            if xrefcd == other_xrefcd or (other_xrefcd in assignment_tracker and assignment_tracker[other_xrefcd] == line_end_data.substation_name):
                                continue  # Skip the same line or lines already assigned to this substation
                            closest_other_line_end_data = None
                            min_distance = float('inf')

                            for other_line_end_list in other_line_ends:
                                for other_line_end_data in other_line_end_list:
                                    # Only consider free line ends
                                    if not other_line_end_data.free_point:
                                        continue
                                    if not BPAGraphCreator.is_name_matching(line_end_data.substation_name, other_xrefcd):
                                        continue
                                    
                                    # Calculate distance between line ends
                                    distance = LineEndDictionary.calc_point_length(line_end_data.point, other_line_end_data.point)
                                    #distance = line_end_data.point.distance(other_line_end_data.point) 
                                    
                                    # Check for proximity and name matching
                                    if distance > proximity :
                                        if DEBUG:
                                            print(f'Warning: LineString {xrefcd} is not close enough to {other_xrefcd}. Distance is {distance} meters')
                                        continue
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_other_line_end_data = other_line_end_data
                                    # Match found, unfree the point and assign the substation
                            if closest_other_line_end_data:
                                if DEBUG:
                                    print(f'Maching substation {line_end_data.substation_name} to {other_xrefcd} at line_idx {closest_other_line_end_data.line_idx} and end_idx {closest_other_line_end_data.end}')
                                closest_other_line_end_data.free_point = False
                                closest_other_line_end_data.substation_name = line_end_data.substation_name
                                closest_other_line_end_data.connection_type = ConnectType.SUBSTATION
                                # Gotta remove the substation from the line_subs_dict
                                if line_end_data.substation_name in self.line_ends_dict.line_subs_dict[other_xrefcd]:
                                    self.line_ends_dict.line_subs_dict[other_xrefcd].remove(line_end_data.substation_name)
                                assignment_tracker[other_xrefcd] = line_end_data.substation_name  # Track this assignment
                                changes_made = True
                                cnt_matches += 1
                        
            # After each iteration, assign substation to remaining free points
            if changes_made:
                if DEBUG:
                    print(f'Total Matching substations: {cnt_matches}')
                    print(f'Total free points: {self.line_ends_dict.total_num_free_points()}')
                self.assignSubstationToRemainingFreePoint()
    ### BUILDING the graph info ######
    @staticmethod
    def strip_name(operating_line_name, voltage):
        """
        Clean line names similar to BPA_Outage_Line_Matcher.strip_name()
        Adapted from BPAOutageMatcher.strip_name() method
        """
        if not operating_line_name or not isinstance(operating_line_name, str):
            return 'UNKNOWN'
            
        tokens = operating_line_name.upper().split() 
        temp = re.sub(r'[^a-zA-Z0-9]+\s*', ' ', " ".join(tokens))  # Replace punctuation with spaces
        return " ".join(temp.split())  # Remove repeated spaces
    
    def _build_graph_info(self):
        """
        Builds a detailed dictionary of graph connections, line features, and cell coverage.
        This is the data that will be saved to the JSON file.
        
        Assumes self.cell_coverage_cache has been populated.
        """
        if self._verbose:
            print("Building graph info dictionary...")
        
        graph_info = OrderedDict()
        
        # 1. Build a helper map of all substation connections
        substation_to_lines_map = defaultdict(set)
        for xrefcd, line_ends_list in self.line_ends_dict.items():
            for line_end_list in line_ends_list:
                for line_end_data in line_end_list:
                    if line_end_data.connection_type == ConnectType.SUBSTATION and \
                       line_end_data.substation_name:
                        substation_to_lines_map[line_end_data.substation_name].add(xrefcd)

        # 2. Iterate through each line in the graph (from line_ends_dict)
        #    and gather all its information.
        
        # Use a set to track processed XRefCds, as self.lines can have duplicates
        processed_xrefcds = set()
        
        for _, row in self.lines.iterrows():
            xrefcd = row['XRefCd']
            
            # Skip lines that were filtered out or already processed
            if xrefcd not in self.line_ends_dict or xrefcd in processed_xrefcds:
                continue
            
            # Get static features from the GeoDataFrame
            operating_line_name = row.get('OperatingLineNm', 'UNKNOWN') # Use .get for safety
            voltage = row.get('VoltageMeas', 0)
            shape_length = row.get('SHAPE_Length', 0)
            cleaned_line_name = BPAGraphCreator.strip_name(operating_line_name, voltage)
            line_type = self._get_line_type(cleaned_line_name)
            
            # Get dynamic features (connections) from line_ends_dict
            substation_connections = []
            line_connections = set()
            my_substations = set()
            
            for line_end_list in self.line_ends_dict[xrefcd]:
                for line_end_data in line_end_list:
                    # Find direct line-to-line connections
                    if line_end_data.connection_type == ConnectType.LINE:
                        line_connections.update(line_end_data.all_merging_lines_xrefcd)
                    
                    # Find this line's substation connections
                    if line_end_data.connection_type == ConnectType.SUBSTATION and \
                       line_end_data.substation_name:
                        my_substations.add(line_end_data.substation_name)
            
            # For each substation this line connects to, find *other* lines at that sub
            for sub_name in my_substations:
                all_lines_at_sub = substation_to_lines_map[sub_name]
                other_lines_at_sub = all_lines_at_sub - {xrefcd}
                substation_connections.append({sub_name: list(other_lines_at_sub)})
            
            # Get cell coverage from the cache
            cells = self.cell_coverage_cache.get(xrefcd, [])
            
            # Store all info
            graph_info[xrefcd] = {
                'voltage': voltage,
                'shape_length': shape_length,
                'line_type': line_type,
                'substation_connections': substation_connections,
                'line_connections': list(line_connections),
                'cell_coverage': cells
            }
            
            processed_xrefcds.add(xrefcd) # Mark as done

        if self._verbose:
            print(f"Built graph info for {len(graph_info)} lines.")
        return graph_info
    #### Collecting the (buffered) coverage of powerlines, for future data collection
    def _get_discretized_line_cells(self, row):
        cells = OrderedDict()
        geom = row['geometry']
        lines = []
        if isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        elif isinstance(geom, LineString):
            lines = [geom]
        else:
            return []
        for line in lines:
            prev_c = None
            for c in line.coords:
                lon = round(c[0], 1)
                lat = round(c[1], 1)
                if prev_c:
                    prev_lon = round(prev_c[0], 1)
                    prev_lat = round(prev_c[1], 1)
                    if abs(lon - prev_lon) > 0.1 or abs(lat - prev_lat) > 0.1:
                        lon_steps = np.linspace(prev_lon, lon, int(abs(lon - prev_lon) / 0.1) + 1)
                        lat_steps = np.linspace(prev_lat, lat, int(abs(lat - prev_lat) / 0.1) + 1)
                        for lon_step, lat_step in zip(lon_steps, lat_steps):
                            cells[(round(lon_step, 1), round(lat_step, 1))] = None
                    else:
                        cells[(lon, lat)] = None
                else:
                    cells[(lon, lat)] = None
                prev_c = c
        return list(cells.keys())

    def _get_buffered_cells(self, row, buffer_km: float):
        geom = row['geometry']
        
        if geom is None or geom.is_empty:
            return []

        metric_crs = self._metric_crs_for_geometry(geom)
        geom_m = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(metric_crs).iloc[0]

        buf_m = float(buffer_km) * 1000.0
        buffered_m = geom_m.buffer(buf_m, cap_style=1, join_style=1)
        #buffered_m = buffered.minimum_rotated_rectangle #We do not need the rectangle, just the buffer

        buffered_wgs = gpd.GeoSeries([buffered_m], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
        minx, miny, maxx, maxy = buffered_wgs.bounds

        def frange(start, stop, step):
            vals = []
            cur = start
            # Normalize to one decimal snap to avoid FP error accumulation
            while cur <= stop + 1e-9:
                vals.append(round(cur, 1))
                cur += step
            return vals

        lon_start = np.floor(minx * 10) / 10.0
        lon_end = np.ceil(maxx * 10) / 10.0
        lat_start = np.floor(miny * 10) / 10.0
        lat_end = np.ceil(maxy * 10) / 10.0

        lons = frange(lon_start, lon_end, 0.1)
        lats = frange(lat_start, lat_end, 0.1)

        if not lons or not lats:
            print("No lons or lats in buffered area.")
            return []  # Empty grid
        
        prepared_buffer = prep(buffered_wgs)

        cells = []
        for lat in lats:
            for lon in lons:
                if prepared_buffer.covers(Point(lon, lat)):
                    cells.append((round(lon, 1), round(lat, 1)))
        return list(OrderedDict.fromkeys(cells).keys())
    
    def _metric_crs_for_geometry(self, geom):
        """
            Gets a suitable UTM (Universal Transverse Mercator) CRS (coordinate system) for a WGS84 geometry.
            This is based on the location and 'zone' where the powerline geometry is located - returns the correct UTM EPSG code.
        """
        centroid = geom.centroid
        lon = centroid.x
        lat = centroid.y
        # Calculate UTM zone
        zone = int((lon + 180) // 6) + 1
        # Determine Northern or Southern hemisphere
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"

    def get_cells_coverage(self, buffer_km=30.0):
        """
        Calculates and caches the cell coverage for all lines in the graph.
        
        On first call, it computes the coverage and stores it in self.cell_coverage_cache.
        On subsequent calls, it returns the aggregated list of cells from the cache.
        
        The cache (self.cell_coverage_cache) is a dictionary:
        { "XRefCd_1": [(lon, lat), ...], "XRefCd_2": [(lon, lat), ...] }
        
        Returns:
            list: A list of all unique (lon, lat) tuples covering all graph lines.
        """
        if self.cell_coverage_cache is not None:
            all_cells = OrderedDict()
            for cells_list in self.cell_coverage_cache.values():
                for cell in cells_list:
                    all_cells[cell] = None
            if self._verbose:
                print(f"Returning cached cell coverage. Unique cells: {len(all_cells.keys())}")
            return list(all_cells.keys())

        # 2. If cache is empty, calculate
        if self._verbose:
            print(f"Calculating cell coverage (buffer: {buffer_km}km) for the first time...")
        
        # Use OrderedDict to build cache, preserving line order
        self.cell_coverage_cache = OrderedDict() 
        total_cells_aggregator = OrderedDict()

        cells = OrderedDict()

        #Note that the 'lines' geojson has multiple entries for some XRefCd; this successfully handles that
        for _, item in self.lines.iterrows():
            xrefcd = item['XRefCd']
            if not(xrefcd in self.line_ends_dict): #If the current xrefcd/line not used, skip it
                continue

            line_cells = self._get_line_cells(item, buffer_km=buffer_km)

            if xrefcd not in self.cell_coverage_cache:
                self.cell_coverage_cache[xrefcd] = OrderedDict()

            for cell in line_cells:
                self.cell_coverage_cache[xrefcd][cell] = None
                total_cells_aggregator[cell] = None

        for xrefcd in self.cell_coverage_cache:
            self.cell_coverage_cache[xrefcd] = list(self.cell_coverage_cache[xrefcd].keys())
        
        if self._verbose:
            print("Unique cells crossed:", len(total_cells_aggregator.keys()))
        
        # Return the aggregated list of all unique cells
        return list(total_cells_aggregator.keys())

    
    def _get_line_cells(self, row, buffer_km=30.0):
        """Return (lon, lat) 0.1° cells within an oriented rectangle around the line.

        If self.buffer_km <= 0, falls back to legacy behavior that samples only the line path.
        """
        if buffer_km and buffer_km > 0:
            try:
                return self._get_buffered_cells(row, buffer_km)
            except Exception as e:
                if self._verbose:
                    print(f"Buffered selection failed: {e}; falling back to legacy discretization.")
                return self._get_discretized_line_cells(row)
        else:
            return self._get_discretized_line_cells(row)
        
    
    def get_connected_lines(self, xrefcd):
        """
        Retrieves lines connected to the given line, separated by connection type.
        
        Args:
            xrefcd (str): The ID of the line.
            
        Returns:
            dict: {'direct': set(xrefcds), 'substation': set(xrefcds)}
        """
        
        # Lazy load a map of Substation -> Set of Lines
        if not hasattr(self, '_substation_line_map'):
            self._substation_line_map = defaultdict(set)
            for x_code, segments in self.line_ends_dict.items():
                for segment in segments:
                    for line_end in segment:
                        if line_end.connection_type == ConnectType.SUBSTATION and line_end.substation_name:
                            self._substation_line_map[line_end.substation_name].add(x_code)

        connections = {'direct': set(), 'substation': set()}
        
        if xrefcd not in self.line_ends_dict:
            return connections

        # Iterate through all ends of the line (start/end of all segments)
        for segment in self.line_ends_dict[xrefcd]:
            for line_end in segment:
                # 1. Direct Physical Connections
                if line_end.connection_type == ConnectType.LINE:
                    connections['direct'].update(line_end.all_merging_lines_xrefcd)
                
                # 2. Substation Connections
                elif line_end.connection_type == ConnectType.SUBSTATION and line_end.substation_name:
                    # Find all other lines at this substation
                    lines_at_sub = self._substation_line_map.get(line_end.substation_name, set())
                    # Exclude self
                    connections['substation'].update(lines_at_sub - {xrefcd})
                    
        return connections