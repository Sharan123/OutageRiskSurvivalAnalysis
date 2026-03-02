# Spatial Processing Code Documentation

---

This document provides an overview of the Python classes used for preprocessing spatial data from BPA (Bonneville Power Administration) and constructing a graph representation of the transmission line network.

The core components of this process are:
1.  `LineEndData`: A data class to store information about the endpoints of transmission line segments.
2.  `LineEndDictionary`: A manager class that organizes and processes `LineEndData` objects, handling initial connections like self-touching and self-intersecting lines.
3.  `BPAGraphCreator`: The main orchestrator class that reads the raw data, applies various rules and heuristics, and uses the other classes to build the final, connected graph structure.

---

## `LineEndData` Class

This class acts as a data container for each endpoint of a transmission line segment. It holds all the necessary attributes to determine its state and connections within the network.

<details>
<summary>Class Attributes</summary>

-   `point`: `shapely.geometry.Point` - The geographic coordinate of the line end.
-   `line_idx`: `int` - The index of the line segment within a `MultiLineString`. It is `None` for `LineString`.
-   `end`: `LineEnd` (Enum: `START` or `END`) - Specifies if it's the start or end point of the line segment.
-   `associated_line_str`: `shapely.geometry.LineString` - The geometry of the line segment this point belongs to,
     which also includes the last (or first) `num_pts`.
-   `free_point`: `bool` - `True` if the endpoint is not yet connected to anything, `False` otherwise.
-   `connection_type`: `ConnectType` (Enum) - The type of connection (e.g., `SUBSTATION`, `LINE`, `SELF`, `FREE`).
-   `substation_name`: `str` - The code (from XRefCd part) of the substation it's connected to, if applicable.
-   `all_merging_lines_xrefcd`: `set` - A set of `XRefCd`s for other lines that connect at this point.

</details>

---

## `LineEndDictionary` Class

This class serves as a specialized dictionary to manage all `LineEndData` objects for the entire dataset. It is initialized with the GeoDataFrame of transmission lines and performs the initial, more deterministic connection steps.

Example of usage:
```python
self.line_ends_dict = LineEndDictionary(self.lines)
print('Total number of free points before filling: ', self.line_ends_dict.total_num_free_points())
self.line_ends_dict.fill_touch_connections()
print('Total number of free points after self-touch: ', self.line_ends_dict.total_num_free_points())
self.line_ends_dict.fill_self_intersection_connections()
print('Total number of free points after self-intersect: ', self.line_ends_dict.total_num_free_points())
self.line_ends_dict._apply_hotfixes_for_substations()
```

<details>
<summary>Key Methods</summary>

-   <details>
    <summary><strong>`__init__(self, lines_gdf)`</strong></summary>
    
    The constructor takes a GeoDataFrame of transmission lines and populates the dictionary by creating `LineEndData` objects for every endpoint of every line segment. It also identifies the substations associated with each line based on its `XRefCd`.
    - Calls `set_unasigned_substations` which for each line splits the `XRefCd` and assigns both substation ends as unassigned. Note: A single line can have more than two substations, however the two ends MUST be assigned. If one or both end are a tap, the last letter `T` is removed. 
    - Calls `init_dict` which: calls `_remove_connected_ends_bool` for each line, this creates the list of line ends (and saves the original idx from `MultiLineString` or `None` for `LineString`), it then checks which ends are matched and marks them as a `SELF` non-free connection. If there are more lines in the geojson with the same `XRefCd`, it adjusts the indexes to avoid clashes when adding to `mapping_dict`. Note: in this case the indexes do not match the original indexes in the geojson `MultiLineString`.
    - `_hotfix_step_up_transformer_connections`: Fixes SUTI (transformer) lines within the substation. Uses `suti_line_rules.txt` to map both XRefCd line parts to correct substation name and assigns the ends as non-free `SUBSTATION` connected lines. Only index `0` and `1` are used. [Warning will be issues if any SUTI line has more than one pair of LineEndData objects which should not happen]. Finally from `line_subs_dict` we discard any SUTI as well as both parts of the renamed lines' XRefCd. E.g. COVI-SUTI-1, COVI, COVI => COVI is removed and SUTI is removed. `ALVI-CAPT-1` is also a SUTI line so it is treated specially. 
    - `_apply_hotfixes_for_self_and_line_connections`: Reads the rules from `reconnect_line_rules.txt`. From there we read `XRefCd`, `line_idx`, `end_type` (`START` or `END`), `connection_type` (e.g. `SELF`, `SUBSTATION`), and `merging_XRefCd` and for each `XRefCd` we update the given `idx_line` in `mapping_dict` and the given end, by updating the `connection_type` and adding the `merging_XRefCd` to the set of merging lines and setting the free point to `False`.
    </details>

-   <details>
    <summary><strong>`fill_touch_connections(self)`</strong></summary>
    
    This crucial method iterates through all line ends and identifies endpoints that are geographically identical ("touching") and free points. It marks these points as connected to each other and assigned them as non-free and sets the connection type to (`ConnectType.LINE`). If the `XRefCd` parts of the matching lines match (based on the rules in `is_xref_cd_matching`) the matching part is assigned as a substation name and the `remove_substation` is called to clear up the `line_subs_dict` (taking into accounts taps, and tap-to-tap connections)
    </details>

-   <details>
    <summary><strong>`fill_self_intersection_connections(self)`</strong></summary>
    
    The goal is to correctly identify which line ends are part of an internal 4-way intersection and mark them as not "free", so they aren't later mistaken for a connection point to another power line or a substation.

    First it collects all points and counts how many line ends touch each point. Then for each point that shares 4 line ends (i.e. 4-way intersection), it separates the line ends into "long" and "short" lines based on their lengths (longer or less than 150m). It then calculates the angles between the long and short lines to confirm they form a proper intersection (i.e., the long lines are roughly perpendicular to the short lines). If the angle checks pass, and if the short lines do not intersect any other `MultiLineString` corresponding to a different `XRefCD` it marks all four line ends of short lines at that point as non-free with a connection type of `ConnectType.SELF`.

    If there are 3 short lines and 1 long line, if checks if there are at least 2 short lines that form an angle of near 90 degrees. If so, it selects the shorter of the two lines and marks the two line ends of that line as non-free with a connection type of `ConnectType.SELF`.
    </details>
    
-   <details>
    <summary><strong>`total_num_free_points(self)`</strong></summary>
    
    Calls `get_num_free_points` for each `XRefCd` in the `mapping_dict` keys and sums up the result. This submethod checks if a line is a `LineString` or `MultiLineString` and counts the number of free endpoints accordingly, since `mapping_dict` will contain either `[start_pt, end_pt]` or `[[start_pt1, end_pt1], [start_pt2, end_pt2], ...]]` respectivelly.
    </details>

-   <details>
    <summary><strong>`_apply_hotfixes_for_substations(self)`</strong></summary>
    
    Reads the rules from `reconnect_substation_rules.txt`. From there we read `XRefCd`, `line_idx`, `end_type` (`START` or `END`), `connection_type` (e.g. `SELF`, `SUBSTATION`) AND `substation_name`. For each such `XRefCd` we update the given `idx_line` in `mapping_dict` and the given end, by updating the `connection_type`, adding the `substation_name` (if applicable), setting the free point to `False`, and removing said substation from `line_subs_dict`.

    The method also applies special `line_subs_dict` removals for particular lines. 
    </details>


    

</details>

<details>
<summary>All Methods</summary>

-   <details>
    <summary>Initialization &amp; Setup</summary>

    - `__init__(self, lines_gdf)`: Constructor. Initializes the dictionary from a GeoDataFrame.
    - `set_debug(self, debug)`: Enables or disables debug print statements.

    </details>

-   <details>
    <summary>Utility &amp; Helper Methods</summary>

    - `total_num_free_points(self)`: Returns the total count of unconnected endpoints across all lines.
    - `get_num_free_points(self, xrefcd)`: Returns the count of unconnected endpoints for a specific line `XRefCd`.
    - `remove_substation(self, xrefcd, substation_name)`: Removes a substation name from a line's list of potential substation connections from `line_subs_dict`, `mapping_dict`, and `self.lines`.
    - `calc_point_length(p1, p2)`: A static method to calculate the distance between two `shapely.geometry.Point` objects.

    </details>

</details>

---

## `BPAGraphCreator` Class

This is the main class that orchestrates the entire graph creation process. It reads the raw GeoJSON data, applies a series of cleaning steps, rule-based heuristics, and proximity-based matching to connect the transmission line endpoints into a coherent network graph.

Example of usage:
```python
bpa_graph = BPAGraphCreator()
bpa_graph.set_debug(True)
bpa_graph.getLineStringIntersects()
bpa_graph.assignSubstationToRemainingFreePoint()
bpa_graph.match_proximity_substations()
```

<details>
<summary>Key Methods</summary>

-   <details>
    <summary><strong>`__init__(self, bpa_geojson_path)`</strong></summary>

    The constructor is the main entry point. It loads the data from the GeoJSON file, applies initial renames using `_apply_renames`, and then initializes the `LineEndDictionary`. After initialization of `LineEndDictionary` the first-pass connection logic is invoked: `fill_touch_connections()`, `fill_self_intersection_connections()`, and `_apply_hotfixes_for_substations()`.
    </details>

-   <details>
    <summary><strong>`getLineStringIntersects(self)`</strong></summary>

    This method connects tap and tap-to-tap lines by finding the intersecting line. It iterates through all free line ends and checks for geometric intersection with other lines. If two lines intersect, it uses a custom naming rule check (`xrefCdIntersectCheck`) to validate if they should be connected (e.g., as a T-junction, or TT-junction). If validated, it updates the line end's `connection_type` to `LINE`, sets `free_point` to `False`, and adds the intersecting line's `XRefCd` to the merging lines set. It makes sure to check which point is closer to the intersection point to assign the correct endpoint (and if it is a single line, and one end is assigned, the other end is skipped). 
    
    It also calls `remove_substation` to remove the 'substation' (which is in fact a line in case of a tap/tap-to-tap) from the `line_subs_dict`.
    </details>

-   <details>
    <summary><strong>`assignSubstationToRemainingFreePoint(self)`</strong></summary>

    A cleanup heuristic. If a line has exactly one remaining free endpoint and one unassigned substation name in its `line_subs_dict`, this method connects them. It sets the endpoint's `connection_type` to `SUBSTATION`, `free_point` to `False`, and assigns the substation name. Usually called after a full setup of `LineEndDictionary` and `getLineStringIntersects` to resolve simple remaining cases.
    </details>

-   <details>
    <summary><strong>`match_proximity_substations(self, proximity=200)`</strong></summary>

    It iteratively matches unconnected ("free") line ends to nearby substations. 
    
    This is done by i) Finding non-free, substation connected points ii) Finding the closest free endpoint of a different line iii) If the line was not already assigned to that substation in the same iteration and if the name is matching to the substation name according to `is_name_matching` (exactly) AND if the distance is less than `proximity` then the free point is assigned this substation and is marked as non-free with adjusted `connection_type`. The process repeats until no more matches can be found in an iteration. After each pass, and if changes are made, it calls `assignSubstationToRemainingFreePoint` to clean up any newly created single-free-point lines.
    </details>

</details>

<details>
<summary>All Methods</summary>

-   <details>
    <summary>Initialization & Data Cleaning</summary>

    -   <code>__init__(self, bpa_geojson_path)</code>: Constructor that loads data and orchestrates the initial processing steps.
    -   `_apply_renames(self)`: Private method to apply line renaming or deletion rules from `rename_rules.txt`.
    -   `set_debug(self, debug)`: Enables or disables debug print statements for this class and its `LineEndDictionary` instance.

    </details>

-   <details>
    <summary>Connection & Matching Logic</summary>

    -   `match_proximity_substations(self, proximity=200)`: Matches free line ends to substations based on proximity and name compatibility.
    -   `assignSubstationToRemainingFreePoint(self)`: Connects the last free point of a line to its last available substation.
    -   `getLineStringIntersects(self)`: Connects lines that intersect mid-span based on geometric intersection and naming conventions.

    </details>

-   <details>
    <summary>Rule & Naming Logic (Static Methods)</summary>

    -   `is_name_matching(substation_name, line_xrefcd)`: Checks if a substation name is compatible with a line's `XRefCd`. It handles tap (`T`) naming conventions by temporarily removing the 'T' for matching purposes.
    -   `xrefCdIntersectCheck(xrefcd, merging_xrefcd)`: If the first part of the xrefcd does not contain 5 letters and the 5th one being 'T' we return `False`. In case it does, we check the second part, which should be 2 letters from first + 2 letters from second `merging_xrefcd`. E.g. `xrefcd=ABERT-SACO`, `merging_xrefcd = SATP-COSM-1` => 'SA+CO=SACO'. If the second part of xrefcd contains 5 letters and last letter is 'T' (tap-to-tap) then we check if that is the case for the first part of the `merging_xrefcd` (if not we return `False`). Otherwise we return `True` if any part of the `xrefcd` matches any part of the `merging_xrefcd` (since tap-to-tap). 

    </details>

</details>