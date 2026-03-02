# Code Documentation

---

<details>
<summary><strong>BPAOutageMatcher</strong></summary>

## 🚀 Quick Start

```python
# Primary workflow - preprocessing new data
outage_match_table = BPAOutageMatcher(verbose=True)
outage_match_table.check_validity()
outage_match_table.preprocess()
outage_match_table.print_unmatched_outage_names()
outage_match_table.grouped.head(n=10)

# Loading preprocessed data
newset = BPAOutageMatcher(verbose=True)
newset.load_csv(filename="preprocessed_outages_2018_2023.csv")
newset.grouped.head(n=10)
```

---

## 🔑 Core Methods (Most Important)

<details>
<summary><strong>🏗️ __init__()</strong></summary>

**Primary constructor for BPAOutageMatcher**

- Loads outages, preprocesses outages, loads line names, and adjusts line names
- Merges the outages across years, gets the number of mismatched patterns (if verbose)
- Finally corrects the naming of lines in `self.combined` by calling `_apply_name_renames`

This is the main entry point that sets up the entire matching system.
</details>

<details>
<summary><strong>⚙️ preprocess()</strong></summary>

**Core processing method that performs the actual matching**

- Applies `outage_line_name_compare` to each row (which effectively matches XRefCd with the name)
- Converts it into a series and drops all rows with XRefCD missing
- If verbose is on, prints in decreasing order the missing lines (per each line type)
- Groups data by: 'Outage ID', 'Voltage (kV)', 'Out Datetime', 'Cause', 'Responsible System'
- Aggregates other fields into lists: 'In Datetime', 'Original Line Name', 'Line Type', 'O&M District', 'Length (miles)', 'Duration (minutes)', 'Name', 'XRefCd', 'XRefCd_Type'
- Finally calls `_remove_duplicates`

This is the main processing pipeline that matches outages to transmission lines.
</details>

<details>
<summary><strong>📊 get_line_features()</strong></summary>

**Extract line features for machine learning models**

- Get line features for a list of XRefCd values using the line information dataframe in `self.line_info`
- Returns a dictionary where keys are xrefcds that matched lines in the line_info dataframe
- Values are lists [voltage, shape_length] + [line_type] where line_type is a one hot encoding of a type (line, tap, or tap-to-tap)

Essential for feature extraction in GNN models.
</details>

<details>
<summary><strong>⏰ time_from_last_outage()</strong></summary>

**Calculate temporal features for outage prediction**

- For a given `xrefcd` and `date` returns a number [0,1] representing the date of the last occurred outage
- Scaled by 90 days in hours (2160). If the outage has not occurred for 90 or more days returns 1

Critical for temporal modeling in outage prediction systems.
</details>

<details>
<summary><strong>💾 load_csv()</strong></summary>

**Load preprocessed outage data**

- Loads the `self.grouped` from a csv file
- The XRefCd and XRefCd_Type are loaded using `correct_and_eval_list_of_lists` that reads a string (e.g. '[L], [T]') and converts it into a list
- Other fields: 'In Datetime','Original Line Name', 'Line Type', 'O&M District', 'Length (miles)', 'Duration (minutes)', 'Name' are loaded into one list (per variable)

Essential for loading previously processed data quickly.
</details>

---

## 📁 Data Loading & Initialization

<details>
<summary><strong>load_outages</strong></summary>

- Loads all outages into `self.dfs_years`. Drops empty columns.
</details>

<details>
<summary><strong>load_line_names</strong></summary>

- Renames 'OperatingLineNm' to 'Name'
- Loads Transmission Lines CSV (names and XRefCd)
- Drops rows with empty fields
</details>

<details>
<summary><strong>adjust_line_names</strong></summary>

- Saves 'Original Line Name'
- Applies `remove_punc_upper` (uppercases, removes non-characters, strips)
</details>

---

## 🔄 Data Processing & Preprocessing

<details>
<summary><strong>preprocess_fields_outage_dataframes</strong></summary>

- Converts "In Datetime" and "Out Datetime" to datetime
- Renames columns
- Fills 'Cause' and 'Responsible System' if corresponding 'Field' columns are present
- Drops 'Field' columns
- Applies `strip_name` to each row (uppercases, checks for LINE(S) and KV #, removes punctuation)
</details>

<details>
<summary><strong>merge_outages</strong></summary>

- Merges all loaded BPA outages from `self.dfs_years`
- Drops planned outages or excluded line types
- Retains lines ending in NO # or NO # #, or those in the exception list
</details>

<details>
<summary><strong>_remove_duplicates</strong></summary>

- Check for duplicates on the 'Outage ID', and if there are any in `self.grouped` it prints them then discards them
- Then prints again (to check that there are no duplicates now)
- Finally prints the head of self.grouped
</details>

---

## 🎯 Matching Engine

<details>
<summary><strong>outage_line_name_compare</strong></summary>

- Gets the `hits`, `hits_type` for the row, based on the row/line's type
- If unknown type, returns np.nan for both
- If no hits was found for the row, the line gets added to `self.no_match_outages[line_type]`
- Finally, two new columns are added to the row - XRefCd and XRefCd_Type, which are populated with `hits` and `hits_type`
</details>

<details>
<summary><strong>find_matching_lines</strong></summary>

- Check if `is_a_line` is true. If it is, checks if the line is of 'NO # #' format - if so, include both numbers NO 1 and NO 2 for example, otherwise include just the one line
- The method searches for the outage recorded name in the transmission lines records
- Exact matches are saved in `hits` while `hits_temp` contains partial matches
- If `self._stop_after_exact_match` is `True` and if we found an exact match - hits is returned, otherwise if `self._stop_after_exact_match` is `False` all partial matches are returned
</details>

<details>
<summary><strong>find_matching_tap</strong></summary>

- If `self._stop_after_exact_match` searches for an exact tap name, otherwise searches for a partial tap name
- If no match is found for a row and if `self._outage_include_both_line_type_ends` is true then performs a line search
</details>

<details>
<summary><strong>find_matching_sections</strong></summary>

- If `self._stop_after_exact_match` searches for an exact section name, otherwise searches for a partial section name
- If no match is found for a row and if `self._outage_include_both_line_type_ends` is true then performs a line search
</details>

<details>
<summary><strong>find_matching_tap_to_tap</strong></summary>

- If `self._stop_after_exact_match` searches for an exact tap to tap name, otherwise searches for a partial tap to tap name
- If no match is found for a row and if `self._outage_include_both_line_type_ends` is true then performs a tap (to a line) search
</details>

<details>
<summary><strong>find_matching_section_of_tap</strong></summary>

- If `self._stop_after_exact_match` searches for an exact section of a tap name, otherwise searches for a partial section of a tap name
- If no match is found for a row and if `self._outage_include_both_line_type_ends` is true then performs a tap search
</details>

---

## ✅ Validation & Type Checking

<details>
<summary><strong>check_validity</strong></summary>

- Ensures data integrity
- Checks that each line's type matches its name (e.g., Tap, Section)
</details>

<details>
<summary><strong>is_a_tap</strong></summary>

- If name contains "TAP TO" exactly once and does not contain "SECTION OF" - it is considered a tap
</details>

<details>
<summary><strong>is_a_line</strong></summary>

- If name contains "TAP TO" or "SECTION OF" - it is not considered a line
</details>

<details>
<summary><strong>is_a_section</strong></summary>

- If name contains "SECTION OF" exactly once and does not contain "TAP TO" - it is considered a section
</details>

<details>
<summary><strong>is_a_tap_to_tap</strong></summary>

- If name contains "TAP TO" exactly twice and does not contain "SECTION OF" - it is considered a tap to tap
</details>

<details>
<summary><strong>is_a_section_of_tap</strong></summary>

- If name contains "SECTION OF" exactly once and "TAP TO" exactly once - it is considered a section of tap
</details>

---

## 🛠️ Utility Methods

<details>
<summary><strong>adjust_line_inclusion_based_on_outage</strong></summary>

- Updates/sets `_outage_include_both_line_type_ends` and `_stop_after_exact_match`
</details>

<details>
<summary><strong>print_unmatched_outage_names</strong></summary>

- Print the names of unmatched lines AFTER `preprocess()` has been run
- Takes into account known non-existing lines and does not print them
</details>

<details>
<summary><strong>save_csv</strong></summary>

- Saves the `self.grouped` to a csv file
- Converts the XRefCd and XRefCd_Type to a list (e.g. '[L], [T]')
- Converts 'In Datetime','Original Line Name', 'Line Type', 'O&M District', 'Length (miles)', 'Duration (minutes)', 'Name' to concatenated strings
</details>

<details>
<summary><strong>_number_of_mismatched_patterns</strong></summary>

- Counts lines not ending in NO # or NO # #
- Checks for known patterns not present in line names data
- Prints unmatched lines not matching "unknown" patterns
</details>

</details>

---

<!-- Add additional classes below using the same collapsible format -->