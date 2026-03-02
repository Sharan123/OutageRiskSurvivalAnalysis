from shapely.geometry import LineString, MultiLineString
from pathlib import Path
import sys
import pyproj
from shapely.ops import transform
import math
from collections import defaultdict
from collections import OrderedDict
import pandas as pd
import pickle

base_dir = Path(__file__).resolve().parent.parent
if base_dir not in sys.path:
    sys.path.append(str(base_dir))

from BPA_Spatial_Preprocessing.LineEndData import LineEndData, ConnectType, LineEnd, DEBUG

class LineEndDictionary:
    def __init__(self, lines):
        self.lines = lines
        self.set_unasigned_substations()
        self.init_dict()
        self._hotfix_step_up_transformer_connections()
        self._apply_hotfixes_for_self_and_line_connections()

    def to_serializable(self, include_lines=False):
        """
            Prepare a serializable payload describing the current mapping and substation state.
        """
        payload = {
            'mapping_dict': self.mapping_dict,
            'line_subs_dict': self.line_subs_dict,
        }
        if include_lines:
            payload['lines'] = self.lines
        return payload

    @classmethod
    def from_serialized(cls, payload, lines=None):
        """
            Rehydrate a LineEndDictionary from a serialized payload produced by to_serializable.
        """
        effective_lines = lines if lines is not None else payload.get('lines')
        if effective_lines is None:
            raise ValueError('lines must be provided either in the payload or via the lines argument.')

        instance = cls.__new__(cls)
        instance.lines = effective_lines
        instance.mapping_dict = payload['mapping_dict']
        instance.line_subs_dict = payload['line_subs_dict']
        return instance

    def save(self, filepath, include_lines=True):
        """
            Persist the dictionary to disk, including the mapping and substation dictionaries.
        """
        path = Path(filepath)
        parent = path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        payload = self.to_serializable(include_lines=include_lines)
        with path.open('wb') as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, filepath, lines=None):
        """
            Load a LineEndDictionary from disk.
        """
        path = Path(filepath)
        with path.open('rb') as file:
            payload = pickle.load(file)
        return cls.from_serialized(payload, lines=lines)

    def set_debug(self, debug=False):
        global DEBUG
        DEBUG = debug
    
    def set_unasigned_substations(self):
        """
            Goes through each line and splits the xrefcd, assigning both substations as unassigned
            Note: A single line can have more than 2 substations assigned
            Sets: self.line_subs_dict - a dictionary with key: xrefcd, value: station_code
            Returns: None
        """
        self.line_subs_dict = {}
        for _, line in self.lines.iterrows():
            xrefcd = line['XRefCd']
            if xrefcd not in self.line_subs_dict:
                self.line_subs_dict[xrefcd] = set()
            xrefcd_split = xrefcd.split('-')[:2]
            # If the first part has 5 letters and ends with T append without T
            if len(xrefcd_split[0]) == 5 and xrefcd_split[0][-1] == 'T':
                xrefcd_split[0] = xrefcd_split[0][:-1]
                if len(xrefcd_split[1]) == 5 and xrefcd_split[1][-1] == 'T':
                    xrefcd_split[1] = xrefcd_split[1][:-1]
            self.line_subs_dict[xrefcd].add(xrefcd_split[0])
            self.line_subs_dict[xrefcd].add(xrefcd_split[1])
    
    def remove_substation(self, xrefcd_from, xrefcd_remove):
        """
           Removes the xrefcd_remove from the xrefcd_from by comparing the substation names. If xrefcd_from has 5 letters and ends with 'T', it treats the 2nd line as if the first merges into that line
           Note: If xrefcd_from has 5 letters and ends with 'T', the substation name is formed by merging the first two letters of the xrefcd_remove's substations
                 If xrefcd_remove also has 5 letters and ends with 'T', the there are no actual substations, but they are formed in the same way
           Updates: self.line_subs_dict
        """
        xrefcd_remove_split = xrefcd_remove.split('-')[:2]
        xrefcd_from_split = xrefcd_from.split('-')[:2]
        connect = False
        if len(xrefcd_from_split[0]) == 5 and xrefcd_from_split[0][-1] == 'T':
            if len(xrefcd_from_split[1]) == 5 and xrefcd_from_split[1][-1] == 'T':
                xrefcd_from_split[0] = xrefcd_from_split[0][:-1]
                xrefcd_from_split[1] = xrefcd_from_split[1][:-1]
            else:
                connect = True
        

        if connect:
                xrefcd_merge_string = xrefcd_remove_split[0][:2] + xrefcd_remove_split[1][:2]
                if xrefcd_merge_string in self.line_subs_dict[xrefcd_from]:
                    self.line_subs_dict[xrefcd_from].remove(xrefcd_merge_string)
        else:
            # check if any of xrefcd_remove_split elements are in line_subs_dict[xrefcd_from] and remove them
            for xrefcd_remove_substation in xrefcd_remove_split:
                if len(xrefcd_remove_substation)==5 and xrefcd_remove_substation[-1]=='T':
                    xrefcd_remove_substation = xrefcd_remove_substation[:-1]
                if xrefcd_remove_substation in self.line_subs_dict[xrefcd_from]:
                    self.line_subs_dict[xrefcd_from].remove(xrefcd_remove_substation)
        
    @staticmethod
    def calculate_angle(line1, line2):
        """
            Calculate the angle between two lines in degrees.
            Note: Lines are unordered, therefore two lines that meet at 180 degrees angle can return either close to 0, or close to 180 degrees, depending on the LineString order
            Returns: Angle between the two lines in degrees
        """
        vector_1 = (line1.coords[1][0] - line1.coords[0][0], line1.coords[1][1] - line1.coords[0][1])
        vector_2 = (line2.coords[1][0] - line2.coords[0][0], line2.coords[1][1] - line2.coords[0][1])
        dot_product = vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]
        magnitude_1 = math.sqrt(vector_1[0]**2 + vector_1[1]**2)
        magnitude_2 = math.sqrt(vector_2[0]**2 + vector_2[1]**2)
        angle_rad = math.acos(dot_product / (magnitude_1 * magnitude_2))
        return math.degrees(angle_rad)

    @staticmethod 
    def calc_point_length(pt1, pt2):
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        pt1_proj = transform(transformer.transform, pt1)
        pt2_proj = transform(transformer.transform, pt2)
        return pt1_proj.distance(pt2_proj)
            
    @staticmethod
    def calculate_length(line, src_crs='EPSG:4326', dest_crs='EPSG:3857'):
        """
            Calculates the length of a LineString in meters. Projects the LineString to the destination CRS using Transformer
            Returns: Length of the line in meters
        """
        transformer = pyproj.Transformer.from_crs(src_crs, dest_crs, always_xy=True)
        line_projected = transform(transformer.transform, line)
        return line_projected.length
    

    @staticmethod
    def _create_ends(line):
        """
            Creates a list of LineEndData objects for each end of the line, for all LineStrings in MultiLineString
            Returns: A list of LineEndData objects for each end of the line
        """
        line_mapping_list = []
        if isinstance(line.geometry, LineString):
            first_pt = LineEndData(None, LineEnd.START, line.geometry, num_pts=2)
            last_pt = LineEndData(None, LineEnd.END, line.geometry, num_pts=2)
            return [first_pt, last_pt]
        elif isinstance(line.geometry, MultiLineString):
            for idx, line_str in enumerate(line.geometry.geoms):
                first_pt = LineEndData(idx, LineEnd.START, line_str, num_pts=2)
                last_pt = LineEndData(idx, LineEnd.END, line_str, num_pts=2)
                line_mapping_list.append([first_pt, last_pt])
        return line_mapping_list
    

    @staticmethod
    def _remove_connected_ends_bool(line):
        """
            Removes the ends of the line which are connected to other lines. 
            Returns: A list of LineEndData objects for each end of the line, for all LineStrings in MultiLineString
        """
        global DEBUG
        ends_list = LineEndDictionary._create_ends(line)
        if isinstance(line.geometry, LineString):
            return ends_list
        elif isinstance(line.geometry, MultiLineString):
            # Check if size is 1, if so return
            if len(line.geometry.geoms) == 1:
                return ends_list
            for i, line_str in enumerate(line.geometry.geoms):
                start_pt = ends_list[i][0].point
                end_pt = ends_list[i][1].point
                for j, other_line_str in enumerate(line.geometry.geoms):
                    if line_str == other_line_str:
                        continue
                    if DEBUG:
                        print(f'Line length at {i} in meters is {LineEndDictionary.calculate_length(line_str)}')
                        print(f'Line length at {j} in meters is {LineEndDictionary.calculate_length(other_line_str)}')
                    # If the line_str intersects with other_line_str, flag it for removal from ends_dict
                    if other_line_str.intersects(start_pt):
                        ends_list[i][0].free_point = False
                        ends_list[i][0].connection_type = ConnectType.SELF
                        if DEBUG:
                            print(f'Removing start point of {i} with value {line_str.coords[0]}')
                    if other_line_str.intersects(end_pt):
                        ends_list[i][1].free_point = False
                        ends_list[i][1].connection_type = ConnectType.SELF
                        if DEBUG:
                            print(f'Removing end point of {i} with value {line_str.coords[-1]}')
            # Remove all the flagged points from ends_dict
            return ends_list
        
    
    @staticmethod
    def _is_near_90_degrees(angle, tolerance=50):
        """
            Check if the angle is close to 90 degrees within a given tolerance.
            Returns: True if the angle is close to 90 degrees, False otherwise
        """
        return 90 - tolerance <= angle <= 90 + tolerance
    
    def check_if_intersects(self, linestr, xrefcd):
        """
            Check if a linestr intersects with any other line from lines, not accounting for self-intersections
            Returns: True if it intersects with any other line from lines, False otherwise
        """
        return any(linestr.intersects(other_line) for other_line in self.lines[self.lines['XRefCd'] != xrefcd].geometry)
      
    def find_intersection_lines_and_points(self, line, line_lineEnds):
        """
            Find all intersection formed by 2 long lines and 2 short lines within a powerline, where intersection lines are at 90 degrees and do not intersect other powerlines.
            Modifies: LineEndData objects in line_lineEnds
            Returns: the modified LineEndData objects in line_lineEnds
        """

        # Step 1: Find connected points with exactly 4 connecting lines
        connected_points = defaultdict(list)
        for idx, linestr in enumerate(line.geometry.geoms):
            start, end = linestr.coords[0], linestr.coords[-1]
            connected_points[start].append(idx)
            connected_points[end].append(idx)
        
        for point, linestr_idxs in connected_points.items():
            if len(linestr_idxs) != 4:
                continue

            if DEBUG:
                print(f'Intersection found at point {point} within the line {line["XRefCd"]} with # lines: {len(linestr_idxs)}')
            # Step 2: Classify lines into long (>150m) and short
            long_lines = []
            short_lines = []
            for idx in linestr_idxs:
                if DEBUG:
                    print(f'Line length at {idx} in meters is {LineEndDictionary.calculate_length(line.geometry.geoms[idx])}')
                if LineEndDictionary.calculate_length(line.geometry.geoms[idx]) > 150:
                    long_lines.append(idx)
                else:
                    short_lines.append(idx)
            
            # Get non-free LineEndData objects for long and short lines
            # shapely.geometric.Point has CoordinateSequence object in coords attribute, so we need to access [0] to get the only point tuple
            long_linestrs = [line_end.associated_line_str for idx in long_lines for line_end in line_lineEnds[idx] if line_end.free_point == False and line_end.point.coords[0] == point]
            short_linestrs = [line_end.associated_line_str for idx in short_lines for line_end in line_lineEnds[idx] if line_end.free_point == False and line_end.point.coords[0] == point]
            # We are asking for associated line str for the 4 lines, at the intersection point (which will give us the 4 intersecting lines for angle calc)
            if DEBUG:
                print('Long lines idxs:', long_lines)
                print('Short Lines idxs:', short_lines)
                print('Angles between long lines:', LineEndDictionary.calculate_angle(long_linestrs[0], long_linestrs[1]))
                print('Angles between short lines:', LineEndDictionary.calculate_angle(short_linestrs[0], short_linestrs[1]))
            if len(long_lines) == 2:
                # Step 3: Check angles between long lines and between the short lines using line_lineEnds at the non-free end points

                if LineEndDictionary._is_near_90_degrees(LineEndDictionary.calculate_angle(long_linestrs[0], long_linestrs[1])) \
                    and LineEndDictionary._is_near_90_degrees(LineEndDictionary.calculate_angle(short_linestrs[0], short_linestrs[1])):
                    
                    # Check that long line forms an angle of close to 90 with one short line
                    angle_checks = [LineEndDictionary._is_near_90_degrees(LineEndDictionary.calculate_angle(long_linestrs[0], short_linestrs[0])),
                                    LineEndDictionary._is_near_90_degrees(LineEndDictionary.calculate_angle(long_linestrs[0], short_linestrs[1]))]
                    if DEBUG:
                        print(f'Angle checks (long,short[0]), (long,short[1]): {angle_checks}')
                    if not any(angle_checks):
                        continue
                    
                    # Step 4: Ensure no short line intersects with any other line from bpa_graph outside the intersection
                    no_external_intersections = all(not self.check_if_intersects(line.geometry.geoms[short_idx], line['XRefCd']) for short_idx in short_lines)
                    
                    
                    if not no_external_intersections:
                            print(f'Short lines: {short_lines} at {line["XRefCd"]} intersect with other lines')
                            continue
                    if DEBUG:
                            print(f'Intersection found at point {point} within the line {line["XRefCd"]}')
                    # Unfree the ends of short lines and set connection type to SELF
                    for short_ln_idx in short_lines:
                        line_lineEnds[short_ln_idx][0].free_point = False
                        line_lineEnds[short_ln_idx][0].connection_type = ConnectType.SELF
                        line_lineEnds[short_ln_idx][1].free_point = False
                        line_lineEnds[short_ln_idx][1].connection_type = ConnectType.SELF
                else:
                    print(f'Warning: Line {line["XRefCd"]} with long {long_lines} short {short_lines} do not form a 90 degree angle')
            elif len(long_lines) == 1:
                # If we have one long line, take the two short perpendicular lines to the long lines, and mark the shorter line end as non-free point 
                
                # Find the short perpendicular lines
                short_perpendicular_lines = []
                for short_idx, short_line in zip(short_lines, short_linestrs):
                    if LineEndDictionary._is_near_90_degrees(LineEndDictionary.calculate_angle(long_linestrs[0], short_line)):
                        if DEBUG:
                            print(f'Line idx {short_idx} is close to 90 degrees with long line')
                        short_perpendicular_lines.append((short_idx, short_line))
                if DEBUG:
                    print(f'Perpendicular short lines: {short_perpendicular_lines}')
                if len(short_perpendicular_lines) != 2:
                    continue
                # Find the shorter line
                shortest_idx, _ = min(short_perpendicular_lines, key=lambda item: LineEndDictionary.calculate_length(item[1]))
                line_lineEnds[shortest_idx][0].free_point = False
                line_lineEnds[shortest_idx][0].connection_type = ConnectType.SELF
                line_lineEnds[shortest_idx][1].free_point = False
                line_lineEnds[shortest_idx][1].connection_type = ConnectType.SELF
            else:
                print(f'Warning: Line {line["XRefCd"]} with long {long_lines} short {short_lines}. Cannot determine pattern!')

        return line_lineEnds
    
    def init_dict(self):
        """
            Initializes the dictionary mapping_dict with key: xrefcd, value: list of LineEndData objects
            Note: If another line has the same xrefcd, the indexes of the LineEndData objects are updated for every subsequent line to match its position in the list
            Sets: self.mapping_dict, a dictionary with key: xrefcd, value: list of LineEndData objects
        """
        self.mapping_dict = {}
        for _, line in self.lines.iterrows():
            xrefcd = line['XRefCd']
            new_list = LineEndDictionary._remove_connected_ends_bool(line)
            if xrefcd not in self.mapping_dict:
                self.mapping_dict[xrefcd] = []
            else:
                # We already had that xrefcd in dict - therefore the indexes are off and they need to be updated within the new_list
                for i, line_end_data in enumerate(new_list):
                    line_end_data[0].line_idx = len(self.mapping_dict[xrefcd]) + i
                    line_end_data[1].line_idx = len(self.mapping_dict[xrefcd]) + i
            self.mapping_dict[xrefcd].extend(new_list)


    def fill_self_intersection_connections(self):
        for xrefcd, line_ends in self.mapping_dict.items():
            line = self.lines[self.lines['XRefCd'] == xrefcd].iloc[0]
            self.find_intersection_lines_and_points(line, line_ends)
    
    def get_num_free_points(self, xrefcd):
        """
            Gets the number of free points for a given xrefcd from self.mapping_dict
            Returns: Number of free points for a given xrefcd
        """
        if xrefcd not in self.mapping_dict:
            raise ValueError(f'{xrefcd} not in mapping_dict')
        # Get line with xrefcd
        line = self.lines[self.lines['XRefCd'] == xrefcd].iloc[0]
        if isinstance(line.geometry, LineString):
            return sum([1 for line_end_data in self.mapping_dict[xrefcd] if line_end_data.free_point])
        elif isinstance(line.geometry, MultiLineString):
            return sum([1 for line_end_data in self.mapping_dict[xrefcd] if line_end_data[LineEnd.START.value].free_point]) + sum([1 for line_end_data in self.mapping_dict[xrefcd] if line_end_data[LineEnd.END.value].free_point])
    
    def total_num_free_points(self):
        """
            Gets the total number of free points for all xrefcds in self.mapping_dict
            Returns: Total number of free points for all xrefcds in self.mapping_dict
        """
        return sum([self.get_num_free_points(xrefcd) for xrefcd in self.mapping_dict.keys()])

    def remove_line(self, xrefcd):
        """
            Removes a line from the this LineEndsDictionary object
            Modifies: self.mapping_dict, self.line_subs_dict
            Returns: None
        """
        if xrefcd in self.mapping_dict:
            del self.mapping_dict[xrefcd]
        if xrefcd in self.line_subs_dict:
            del self.line_subs_dict[xrefcd]
        self.lines = self.lines[self.lines['XRefCd'] != xrefcd]
    
    def items(self):
        return iter(self.mapping_dict.items())
    def __getitem__(self, key):
        return self.mapping_dict[key]
    def __contains__(self, key):
        return key in self.mapping_dict
    def __setitem__(self, key, value):
        self.mapping_dict[key] = value
    def keys(self):
        return self.mapping_dict.keys()
    
    # Move to graph or create a DatasetManager class which is responsible for the dataset 
    # And then the DataLoader class (which extends pytorch Dataset) which ideally we can use for the Graph samples creation?
    def find_inoperable_xrefcds(self, check_datetime):
        # Convert check_datetime into a pandas datetime object if it's not already
        check_datetime = pd.to_datetime(check_datetime)
        
        # Initialize an empty list to store inoperable XRefCds
        inoperable_xrefcds = OrderedDict()

        # Iterate through each row in the dataframe
        for index, row in self.grouped.iterrows():
            out_datetime = pd.to_datetime(row['Out Datetime'])
            # Iterate over each group of XRefCds and their corresponding in_datetimes
            for xrefcd_group, in_datetime in zip(row['XRefCd'], row['In Datetime']):
                # Convert each in_datetime in the group to datetime object
                # There might be multiple in_datetimes for a group, handling it by iterating through them
                in_datetime = pd.to_datetime(in_datetime)
                # Check if the datetime to check is within the outage period
                if out_datetime <= check_datetime <= in_datetime:
                    # Add all XRefCds in this group to the list of inoperable XRefCds, if not already added
                    for xrefcd in xrefcd_group:
                        if xrefcd not in inoperable_xrefcds:
                            inoperable_xrefcds[xrefcd] = None

        return list(inoperable_xrefcds.keys())
    

    def _hotfix_step_up_transformer_connections(self):
        with open(Path(__file__).parent /'suti_line_rules.txt', 'r') as file:
            transformer_rules = file.readlines()

        # Iterate through each rule in the file
        for rule in transformer_rules:
            # Ignore empty lines
            if not rule.strip():
                continue

            # Split the rule into components
            parts = rule.strip().split(',')
            if len(parts) != 3:  # Ensure there are exactly three parts
                print(f'ERROR: Invalid rule: {rule}')
                continue
            xrefcd, substation_start, substation_end = parts

            # Trim whitespace around components
            xrefcd = xrefcd.strip()
            substation_start = substation_start.strip()
            substation_end = substation_end.strip()

            # Update the substation names for the specified line in line_ends_dict
            if xrefcd in self.mapping_dict:
                if len(self.mapping_dict[xrefcd]) > 1:
                    print(f'WARNING: Line {xrefcd} has more than one pair of LineEndData objects. Applying SUTI hotfix to the first one only.')
                # Assume the rule applies to the first (0th) LineEndData objects for start and end
                self.mapping_dict[xrefcd][0][0].substation_name = substation_start
                self.mapping_dict[xrefcd][0][1].substation_name = substation_end
                self.mapping_dict[xrefcd][0][0].free_point = False
                self.mapping_dict[xrefcd][0][1].free_point = False
                self.mapping_dict[xrefcd][0][0].connection_type = ConnectType.SUBSTATION
                self.mapping_dict[xrefcd][0][1].connection_type = ConnectType.SUBSTATION
                if xrefcd in self.line_subs_dict:
                    self.line_subs_dict[xrefcd].discard(substation_start)
                if xrefcd in self.line_subs_dict:
                    self.line_subs_dict[xrefcd].discard(substation_end)
            else:
                print(f'WARNING: Line {xrefcd} not found in line_ends_dict.')

            # Remove the SUTI 'substation' from the line_subs_dict for the specified line
            if xrefcd in self.line_subs_dict:
                self.line_subs_dict[xrefcd].discard('SUTI')
            else:
                print(f'WARNING: Line {xrefcd} not found in line_subs_dict.')

        if 'ALVY-CAPT-1' in self.line_subs_dict:
            self.line_subs_dict['ALVY-CAPT-1'].discard('CAPT')

    def _apply_hotfixes_for_self_and_line_connections(self):
        """
            Applies hotfixes for self and line connections for specific lines. Reads the rules from reconnect_line_rules.txt
            Note: The hotfixes are applied to the line_ends_dict
            Returns: None
        """
        with open(Path(__file__).parent / 'reconnect_line_rules.txt', 'r') as file:
            hotfix_rules = file.readlines()

        # Iterate through each rule in the file
        for rule in hotfix_rules:
            # Ignore comments and empty lines
            rule = rule.split(';')[0].strip()  # Remove comments and strip whitespace
            if not rule:
                continue
            # Extract rule components
            parts = rule.split(',')
            if len(parts) < 4:  # Ensure there are enough parts to form a rule
                print(f'ERROR: Invalid rule: {rule}')
                continue
            
            xrefcd, idx_line, idx_end_pt, connection_type_str, merging_xrefcd = [part.strip() for part in parts]
            idx_line = int(idx_line)
            idx_end_pt = int(idx_end_pt)  
            # Determine the connection type
            if connection_type_str == 'Line':
                connection_type = ConnectType.LINE
            elif connection_type_str == 'Self':
                connection_type = ConnectType.SELF
            else:
                print(f'ERROR: Invalid connection type: {connection_type_str}')
                continue  # Skip invalid connection types
             # Apply the hotfix to the line_end_data in the line_ends_dict
            if xrefcd in self.mapping_dict:
                # Get the specific line_end_data object
                line_end_data = self.mapping_dict[xrefcd][idx_line][idx_end_pt]

                # Update the properties based on the rule
                line_end_data.free_point = False
                line_end_data.connection_type = connection_type
                if connection_type == ConnectType.LINE and merging_xrefcd != 'None':
                    line_end_data.all_merging_lines_xrefcd.add(merging_xrefcd)
                elif connection_type == ConnectType.SELF:
                    # Clear the substations
                    line_end_data.substation_name = None

    def fill_touch_connections(self):
        """
            For each line, checks if the ends of that line touch any other end of any line in self.mapping_dict, and if so, assigns the connection type to LINE. Sets that end to non-free and fills the merging_lines_xrefcd
            Returns: The modified LineEndData objects in line_lineEnds
        """
        # Iterate over each line by XRefCd
        for xrefcd, line_ends in self.items():
            # For each line, check both start and end points for touches
            for line_end_list in line_ends:
                for line_end_data in line_end_list:
                    # Skip if the point is not free
                    if not line_end_data.free_point:
                        continue
                   
                    touching_lines_xrefcds = []
                    # Check if this line end touches any other line
                    for other_xrefcd, other_line_ends in self.items():
                        if xrefcd == other_xrefcd:
                            continue  # Skip self-check
                        found_in_line = False
                        for other_line_end_list in other_line_ends:
                            for other_line_end_data in other_line_end_list:
                                # Check for touch
                                if line_end_data.point.equals(other_line_end_data.point):
                                    if DEBUG:
                                        print(f'LineString {xrefcd} touches {other_xrefcd}')
                                    touching_lines_xrefcds.append(other_xrefcd)
                                    found_in_line = True
                                    matching_part =  LineEndDictionary.is_xrefcd_matching(xrefcd, other_xrefcd)
                                    if matching_part:
                                        line_end_data.substation_name = matching_part
                                        self.remove_substation(xrefcd, other_xrefcd)
                                    # Once a touch is found, no need to check other ends of this line
                                    break
                            if found_in_line:
                                break
                    if touching_lines_xrefcds:
                        line_end_data.connection_type = ConnectType.LINE
                        line_end_data.free_point = False
                        # Extend the all_merging_lines_xrefcd list with all touching lines
                        line_end_data.all_merging_lines_xrefcd.update(touching_lines_xrefcds)
    
    @staticmethod
    def is_xrefcd_matching(line1_xrefcd, line2_xrefcd):
        """
            Checks if any substation part of line1 xrefcd matches any substation part of line2 xrefcd
            Returns: True if they match, False otherwise
        """
        line1_parts = line1_xrefcd.split('-')
        line2_parts = line2_xrefcd.split('-')
        for part1 in line1_parts[:2]:
            for part2 in line2_parts[:2]:
                if part1 == part2:
                    return part1
        return None
    
    def _apply_hotfixes_for_substations(self):
        
        with open(Path(__file__).parent /'reconnect_substation_rules.txt', 'r') as file:
            hotfix_rules = file.readlines()

        # Iterate through each rule in the file
        for rule in hotfix_rules:
            # Ignore comments and empty lines
            rule = rule.split(';')[0].strip()  # Remove comments and strip whitespace
            if not rule:
                continue

            # Extract rule components
            parts = rule.split(',')
            if len(parts) < 4:  # Ensure there are enough parts to form a rule
                print(f'ERROR: Invalid rule: {rule}')
                continue
            xrefcd, idx_line, idx_end_pt, connection_type_str, substation_code = [part.strip() for part in parts]
            idx_line = int(idx_line)
            idx_end_pt = int(idx_end_pt)

            if connection_type_str in ['Substation', 'Self', 'None']:
                if connection_type_str == 'Substation':
                    connection_type = ConnectType.SUBSTATION
                elif connection_type_str == 'Self':
                    connection_type = ConnectType.SELF
                else:
                    connection_type = None  # This will indicate not to change the existing connection type
            else:
                print(f'ERROR: Invalid connection type: {connection_type_str}')
                continue  # Skip invalid connection types
            
            # Apply the hotfix to the line_end_data in the line_ends_dict
            if xrefcd in self.mapping_dict:
                # Get the specific line_end_data object
                line_end_data = self.mapping_dict[xrefcd][idx_line][idx_end_pt]
                # Determine the connection type
                if connection_type is not None:
                    line_end_data.free_point = False
                    line_end_data.connection_type = connection_type
                if connection_type == ConnectType.SUBSTATION or connection_type is None:
                    line_end_data.substation_name = substation_code
                    # Directly remove the substation from self.line_subs_dict if present
                    if substation_code in self.line_subs_dict[xrefcd]:
                        self.line_subs_dict[xrefcd].remove(substation_code)
                
        if 'LUHV-1' in self.line_subs_dict and '1' in self.line_subs_dict['LUHV-1']:
            self.line_subs_dict['LUHV-1'].remove('1')
        if 'SHKT2-EMTI-1' in self.line_subs_dict and 'SHKT2' in self.line_subs_dict['SHKT2-EMTI-1']:
            self.line_subs_dict['SHKT2-EMTI-1'].remove('SHKT2')
        if 'SHKT2-EMTI-2' in self.line_subs_dict and 'SHKT2' in self.line_subs_dict['SHKT2-EMTI-2']:
            self.line_subs_dict['SHKT2-EMTI-2'].remove('SHKT2')  
        if 'MVERT-LKAL-2' in self.line_subs_dict and 'LKAL' in self.line_subs_dict['MVERT-LKAL-2']:
            self.line_subs_dict['MVERT-LKAL-2'].remove('LKAL')
        if 'HEDGT-FRBC-2' in self.line_subs_dict:
            self.line_subs_dict['HEDGT-FRBC-2'].clear()
