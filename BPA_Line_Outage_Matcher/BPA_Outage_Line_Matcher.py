import pandas as pd
import string
import numpy as np
import re
from collections import Counter
import copy
from tabulate import tabulate
import ast
import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
if base_dir not in sys.path:
    sys.path.append(str(base_dir))

class BPAOutageMatcher:
    def __init__(self, years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"],
                        excluded_line_types = ['S', 'TS'],
                        verbose=False) -> None:
        """
        Contructor for BPAOutageMatcher.
            years: list of years to load data for
            excluded_line_types: list of line types to exclude from the matching process
            outage_include_both_line_type_ends: If True, in a TT, TS, T, S scenario - if we cannot find the first tap, we include the second in the XRefCd list (similar for T,S, and lines)
        Return: None
        """
        self._excluded_line_types = excluded_line_types 
        # Given that the full list of line_types is ['L', 'T', 'TT', 'TS', 'S'] create a list on non-excluded line_types based on excluded ones
        self.line_types = [lt for lt in ['T', 'S', 'TT', 'TS', 'L'] if lt not in excluded_line_types]
        self._outage_include_both_line_type_ends = False
        self._stop_after_exact_match = True
        self._verbose = verbose
        self.load_outages(years)
        self.preprocess_fields_outage_dataframes()
        self.load_line_names()
        self._apply_xref_renames()
        self.adjust_line_names()
        self.merge_outages()

        if self._verbose:
            self._number_of_mismatched_patterns()

        self._apply_name_renames()

    def adjust_line_inclusion_based_on_outage(self, outage_include_both_line_type_ends, stop_after_exact_match):
        """
            outage_include_both_line_type_ends: If True, in a TT, TS, T, S scenario - if we cannot find the first tap, we include the second in the XRefCd list (similar for T,S, and lines)
            stop_after_exact_match: If True, we stop searching for a match after the first exact match is found
        """
        self._outage_include_both_line_type_ends = outage_include_both_line_type_ends
        self._stop_after_exact_match = stop_after_exact_match

    def _number_of_mismatched_patterns(self):
        ctr=0
        for _, row in self.combined.iterrows():
            if not re.search(r'NO \d+( \d+)?$', row['Name']):
                #print(row['Name'])
                ctr+=1  
        print(ctr)

        ctr=0
        names = []
        for index, row in self.combined.iterrows():
            if not re.search(r'NO \d+( \d+)?$', row['Name']):
                # Known non-existing matches
                if re.search(r'LAKES|SUNSET|REFINERY|MEEKER|WEYCO|GLENOMA|FOSSIL MAUPIN|'
                            r'OKANOGAN TWISP|BREW|WILLAM|EWE|119|MULLAN|PHA|CIN|TWISP OKANOGAN|'
                            r'USK PINE|WHITE CREEK|TACOMA TPWR|KELS|LAU|WILLI|7TH|CUR|BOX|OLIV|SHAW|RUNY|'
                            r'HOOP|SANTA|CHERR|DIL|USK USK|WILLAK|COLD|HAZ|RIVER ROAD|WOODL|WILLOW CREEK LANE|RATH|'
                            r'PACW WALLULA|MERW|CEDARVILLE JUNCTION TAP TO MILE HI ALTURAS|CARR|LINDE|'
                            r'MCNARY WALLULA|MILE HI ALTURAS|' # Missing lines with present taps to them
                            r'HAT ROCK TAP TO MCNARY WALLULA|DAVIS CREEK TAP TO MILE HI ALTURAS|'
                            r'KALAMA|SENECA|DELTA', row['Name']):
                    continue
                names.append(row['Name'])
                ctr+=1  
        print('Num of patterns that do no match name # (#)? is: ', ctr)
        # Sort names by the number of appearance, highest to lowest
        name_counts = Counter(names)
        names = sorted(set(names), key=lambda x: (-name_counts[x], names.index(x)))
        print(names)

    
    def load_outages(self, years):
        if self._verbose:
            # Print the working directory
            print(os.getcwd())
        self.dfs_years = [] #To store the dataframes after loading
        filename = "Datasets/BPA_trm/OutagesCY{year}-TransmissionLineInterruptions.csv" 
        for year in years:
            data = pd.read_csv(Path(__file__).resolve().parent.parent/filename.format(year=year), header=1)
            data.name = year
            data.dropna(how='all', axis='columns', inplace=True) #Remove completely empty columns
            #https://stackoverflow.com/questions/43983622/remove-unnamed-columns-in-pandas-dataframe
            self.dfs_years.append(data)
    
    def preprocess_fields_outage_dataframes(self):
        for df_year in self.dfs_years:
            df_year["Out Datetime"] = pd.to_datetime(df_year["Out Datetime"])
            df_year["In Datetime"] = pd.to_datetime(df_year["In Datetime"])
            #Rename "Responsible System Dispatch" and "Cause Dispatch", and District to be consistant. Save original line names 
            df_year.rename(columns={"Responsible System Dispatch": "Responsible System", "Cause Dispatch" : "Cause", 
                                    "District": "O&M District", "Name": "Original Line Name"},inplace=True)

            df_year["Name"] = df_year.apply(BPAOutageMatcher.strip_name, axis=1)

            # For each row in df_year dataframe, if the columns 'Cause Field' and 'Responsible System Field' exist
            # check if a value is present under the 'Cause Field' column. If so, replace the value in the 'Cause' column with the value in the 'Cause Field' column.
            if 'Cause Field' in df_year.columns and 'Responsible System Field' in df_year.columns:
                for index, row in df_year.iterrows():
                    if pd.notnull(row['Cause Field']):
                        df_year.loc[index, 'Cause'] = row['Cause Field']
                    if pd.notnull(row['Responsible System Field']):
                        df_year.loc[index, 'Responsible System'] = row['Responsible System Field']
    
            #Remove the "Cause Field" and "Responsible Systems Field" column    
            if "Cause Field" in df_year.columns:
                df_year.drop(columns=["Cause Field"], inplace=True)
            if "Responsible System Field" in df_year.columns:
                df_year.drop(columns=["Responsible System Field"], inplace=True)
    
    def merge_outages(self):
        exception_list = ['HAT ROCK TAP TO MCNARY WALLULA', 'DAVIS CREEK TAP TO MILE HI ALTURAS', 'ANTELOPE LOST RIVER']

        self.combined = pd.concat(self.dfs_years, join='inner', ignore_index=True)
        # Remove all rows where 'Outage Type' is 'Plan' and rows where Line Type is in the excluded_line_types list
        self.combined = self.combined.drop(self.combined[(self.combined['Outage Type'] == 'Plan') | (self.combined['Line Type'].isin(self._excluded_line_types))].index)
        self.combined.sort_values(by=["Outage ID"], inplace=True, kind='stable', ignore_index=True)
        # Remove all rows whose names do not end in NO # or NO # # where # is a number, except the names in the exception_list
        self.combined = self.combined[self.combined['Name'].isin(exception_list) | self.combined['Name'].str.contains(r'NO \d+(?: \d+)?$')]
    
    @staticmethod
    def strip_name(row):
        tokens = row["Original Line Name"].upper().split() 
        
        if( (not(tokens[-1] == "LINE" or tokens[-1] == "LINES"))
        or (float(tokens[-2].replace("KV","")) != row["Voltage (kV)"]) ): 
            print("Problem! ", row)

        #New name is old name without "###KV LINE" or any punctuation
        temp = " ".join(tokens[0:-2]) 
        temp = re.sub(r'[^a-zA-Z0-9]+\s*', ' ', temp) #Replace punctuation with spaces
        return " ".join(temp.split())
    
    @staticmethod
    def remove_punc_upper(row):
        temp = row["Name"].upper() 
        temp = re.sub(r'[^a-zA-Z0-9]+\s*', ' ', temp) #Replace punctuation with spaces
        return " ".join(temp.split()) #Remove repeated spaces
        
    def load_line_names(self):
        self.line_info = pd.read_csv(Path(__file__).resolve().parent.parent/"Datasets/TransmissionLineData/BPA_TransmissionLines.csv")
        self.line_info.rename(columns={"OperatingLineNm": "Name"},inplace=True)
        self.line_info.dropna(inplace=True) # Drop HLTP-WRNR-1 without a name (there is a duplicate that has a name)
    
    def adjust_line_names(self):
        self.line_info["Original Line Name"] = self.line_info.apply(lambda row: row["Name"], axis=1)
        #The clean the old name
        self.line_info["Name"] = self.line_info.apply(BPAOutageMatcher.remove_punc_upper, axis=1)

    @staticmethod
    def is_a_tap(name_outage):
        # Check if name_outage has a ' TAP TO ' in it and obtain left and right side of the string
        match = re.match(r'.* TAP TO .*', name_outage)
        if not match or name_outage.count('TAP TO')!=1 or name_outage.count('SECTION OF')!=0:
            return False
        return True
    
    @staticmethod
    def is_a_line(name_outage):
        if 'SECTION OF' in name_outage or 'TAP TO' in name_outage:
            return False
        return True

    @staticmethod
    def is_a_tap_to_tap(name_outage):
        match = re.match(r'.* TAP TO .* TAP TO .*', name_outage)
        if not match or name_outage.count('TAP TO')!=2 or name_outage.count('SECTION OF')!=0:
            return False
        return True

    @staticmethod
    def is_a_section_of_tap(name_outage): # TODO CHECK, currently unused
        match = re.match(r'.* SECTION OF .* TAP TO .*', name_outage)
        if not match or name_outage.count('TAP TO')!=1 or name_outage.count('SECTION OF')!=1:
            return False
        return True
    
    @staticmethod
    def is_a_section(name_outage): # TODO CHECK, currently unused
        if name_outage.count('SECTION OF')!=1 or name_outage.count('TAP TO')!=0:
            return False
        return True
    
    @staticmethod
    def check_tap_match(n, line_name, tap_name):
        # Using regular expression check if n is in a form of '... tap_name... TAP TO ... line_name ...' where ... is any string of characters and spaces
        if not BPAOutageMatcher.is_a_tap(n):
            return False
        # Check if exact match exists in form of tap_name TAP TO line_name
        match = re.search(r'.*'+tap_name+'.* TAP TO .*'+line_name+'.*', n)
        if match:
            return True
        return False

    @staticmethod
    def check_line_match(n, line_name):
        # Using regular expression check if n is in a form of '... line_name ...' where ... is any string of characters and spaces, AND n does not contain 'TAP TO'
        if not BPAOutageMatcher.is_a_line(n):
            return False
        match = re.search(r'.*'+line_name+'.*', n)
        if match:
            return True
        return False
    
    @staticmethod
    def check_tap_to_tap_match(n, tap1, tap2, line_name):
        # We are performing a slightly different check, there is no tap to tap in df_lines so we will perform the tap to check (tap1 to line_name) as well as tap1 tap to tap2
        # as well as the tap to tap to line check in case the dataset gets changed or expanded
        match1 = re.match(r'.*'+tap1+'.* TAP TO .*'+tap2+'.*', n)
        match2 = re.match(r'.*'+tap1+'.* TAP TO .*'+line_name+'.*', n)
        match3 = re.match(r'.*'+tap1+'.* TAP TO .*'+tap2+'.* TAP TO .*'+line_name+'.*', n)
        if match3:
            return True
        elif match1 or match2:
            print("Partial match found for", n)
            return True
        return False
    
    @staticmethod
    def check_section_of_tap_match(n, section, tap, line): # TODO CHECK WHEN EXTENDING, unused in this version
        match_full = re.match(r'.*'+section+'.* SECTION OF .*'+tap+'.* TAP TO .*'+line+'.*', n)
        match_section_tap = re.match(r'.*'+section+'.* SECTION OF .*'+tap+'.*', n)
        match_section_line = re.match(r'.*'+section+'.* SECTION OF .*'+line+'.*', n)
        if match_full:
            return True
        elif match_section_tap or match_section_line:
            print("Partial match found for", n)
            return True
        return False
    
    @staticmethod
    def check_section_match(n, section, line): # TODO CHECK WHEN EXTENDING, unused in this version
        if not BPAOutageMatcher.is_a_section(n):
            return False
        match_full = re.match(r'.*'+section+'.* SECTION OF '+line+'.*', n)
        if match_full:
            return True
        return False
    
    @staticmethod
    def check_tap_match_exact(n, line_name, tap_name):
        # Using regular expression check if n is in a form of '... tap_name... TAP TO ... line_name ...' where ... is any string of characters and spaces
        if not BPAOutageMatcher.is_a_tap(n):
            return False
        # Check if exact match exists in form of tap_name TAP TO line_name
        match = re.search(r'^'+tap_name+' TAP TO '+line_name+'$', n)
        if match:
            return True
        return False

    @staticmethod
    def check_line_match_exact(n, line_name):
        # Using regular expression check if n is in a form of '... line_name ...' where ... is any string of characters and spaces, AND n does not contain 'TAP TO'
        if not BPAOutageMatcher.is_a_line(n):
            return False
        match = re.search(r'^'+line_name+'$', n)
        if match:
            return True
        return False

    @staticmethod
    def check_tap_to_tap_match_exact(n, tap1, tap2, line_name):
        # We are performing a slightly different check, there is no tap to tap in df_lines so we will perform the tap to check (tap1 to line_name) as well as tap1 tap to tap2
        # as well as the tap to tap to line check in case the dataset gets changed or expanded
        match1 = re.match(r'^'+tap1+' TAP TO '+tap2+'$', n)
        match2 = re.match(r'^'+tap1+' TAP TO '+line_name+'$', n)
        match3 = re.match(r'^'+tap1+' TAP TO '+tap2+' TAP TO '+line_name+'$', n)
        if match3:
            return True
        elif match1 or match2:
            print("Partial match found for", n)
            return True
        return False

    @staticmethod
    def check_section_of_tap_match_exact(n, section, tap, line): # TODO CHECK WHEN EXTENDING
        match_full = re.match(r'^'+section+' SECTION OF '+tap+' TAP TO '+line+'$', n)
        match_section_tap = re.match(r'^'+section+' SECTION OF '+tap+'$', n)
        match_section_line = re.match(r'^'+section+' SECTION OF '+line+'$', n)
        if match_full:
            return True
        elif match_section_tap or match_section_line:
            print("Partial match found for", n)
            return True
        return False

    @staticmethod
    def check_section_match_exact(n, section, line): # TODO CHECK WHEN EXTENDING
        if not BPAOutageMatcher.is_a_section(n):
            return False
        match_full = re.match(r'^'+section+' SECTION OF '+line+'$', n)
        if match_full:
            return True
        return False
    
    def _apply_name_renames(self):
        with open(Path(__file__).resolve().parent.parent/'BPA_Line_Outage_Matcher/rename_name_rules.txt', 'r') as f:
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
            
            if ('TAP TO' in original_name) and not ('TAP TO' in new_name):
                self.combined.loc[self.combined['Name'] == original_name, 'Line Type'] = 'L'
            if 'TIE TO' in original_name:
                self.combined.loc[self.combined['Name'] == original_name, 'Line Type'] = 'L' # Tricky example it's a TAP but we treat it as a Line for search purposes

            # Update the XRefCd of the line in self.lines
            self.combined.loc[self.combined['Name'] == original_name, 'Name'] = new_name
        if self._verbose:
            print(tabulate(self.combined.head(), headers="keys", tablefmt="psql"))

    def _apply_xref_renames(self):
        """
        Applies XRefCd renaming rules from BPA_Spatial_Preprocessing/rename_rules.txt
        to match the graph structure.
        """
        rename_path = Path(__file__).resolve().parent.parent / 'BPA_Spatial_Preprocessing/rename_rules.txt'
        if not rename_path.exists():
            if self._verbose:
                print(f"Warning: Rename rules not found at {rename_path}")
            return

        with open(rename_path, 'r') as f:
            rename_rules = f.readlines()

        for rule in rename_rules:
            rule = rule.split(';')[0].strip()
            if not rule:
                continue

            original_xref, new_xref = rule.split('->')
            original_xref = original_xref.strip()
            new_xref = new_xref.strip()

            if new_xref.upper() == 'X':
                self.line_info = self.line_info[self.line_info['XRefCd'] != original_xref]
            else:
                self.line_info.loc[self.line_info['XRefCd'] == original_xref, 'XRefCd'] = new_xref



    def find_matching_lines(self, row):
        line_name = row['Name']
        hits = []
        if not BPAOutageMatcher.is_a_line(line_name):
            print("Warning: Line name contains tap or section triggers. Line name:", line_name, 
                "Out Datetime:", row['Out Datetime'], "In Datetime:", row['In Datetime'])
            return [], []
        match = re.match(r'(.*) NO (\d+) (\d+)', line_name)
        lines = []
        if match: #Multiline match
            line_name = match.group(1)
            line1_num = match.group(2)
            line2_num = match.group(3)
            lines = [line_name + ' ' + line1_num + ' ' + line2_num + ' NO ' + line1_num,
                    line_name + ' ' + line1_num + ' ' + line2_num + ' NO ' + line2_num]
        else:
            line_name = row['Name']
            lines = [line_name]
        for ln in lines:
            hits_temp = copy.deepcopy(hits)
            exact_match_found = False
            for i, n in enumerate(self.line_info["Name"]):
            # Check if ANY line_name in lines mathes n, based on check_line_match test
                if self._stop_after_exact_match and BPAOutageMatcher.check_line_match_exact(n, ln):
                    hits.append(self.line_info.iloc[i]['XRefCd'])
                    exact_match_found = True
                    break
                if BPAOutageMatcher.check_line_match(n, ln):
                    hits_temp.append(self.line_info.iloc[i]['XRefCd'])
            if not exact_match_found:
                hits = hits_temp
        return hits, ['L']*len(hits)
    
    def find_matching_tap(self, row):
        # Check that name is in form of '... tap name ... TAP TO ... line name ...' where ... is any string of characters and spaces and there are no more than one 'TAP TO', and none of 'SECTION OF'
        name = row['Name']
        hits = []
        if not BPAOutageMatcher.is_a_tap(name):
            print("Warning: Tap contains multiple tap or section triggers.", name, 
                "Out Datetime:", row['Out Datetime'], "In Datetime:", row['In Datetime'])
            return [], []
        match = re.match(r'(.*) TAP TO (.*)', name)
        tap_name = match.group(1)
        line_name = match.group(2)
        # Go through df_lines and check for rows that contain '...tap_name... TAP TO ...line name...'
        for i, n in enumerate(self.line_info['Name']):
            if self._stop_after_exact_match and BPAOutageMatcher.check_tap_match_exact(n, line_name, tap_name):
                return [self.line_info.iloc[i]['XRefCd']], ['T']
            if BPAOutageMatcher.check_tap_match(n, line_name, tap_name):
                hits.append(self.line_info.iloc[i]['XRefCd'])
        # check if hits is empty
        if not hits and self._outage_include_both_line_type_ends:
            new_row = row.copy()
            new_row['Name'] = line_name
            new_row['Line Type']='L'
            return self.find_matching_lines(new_row)
        return hits, ['T']*len(hits)

    def find_matching_tap_to_tap(self, row):
        hits = []
        name = row['Name']
        if not BPAOutageMatcher.is_a_tap_to_tap(name):
            print("Warning: Tap to Tap contains invalid triggers.", name, 
                "Out Datetime:", row['Out Datetime'], "In Datetime:", row['In Datetime'])
            return [], []
        match = re.match(r'(.*) TAP TO (.*) TAP TO (.*)', name)
        tap1 = match.group(1)
        tap2 = match.group(2)
        line = match.group(3)
        for i,n in enumerate(self.line_info['Name']):
            if self._stop_after_exact_match and BPAOutageMatcher.check_tap_to_tap_match_exact(n, tap1, tap2, line):
                return [self.line_info.iloc[i]['XRefCd']], ['TT']
            if BPAOutageMatcher.check_tap_to_tap_match(n, tap1, tap2, line):
                hits.append(self.line_info.iloc[i]['XRefCd'])
        if not hits and self._outage_include_both_line_type_ends:
            new_row = row.copy()
            new_row['Name'] = tap2+' TAP TO '+line
            new_row['Line Type'] = 'T'
            return self.find_matching_tap(new_row)
        return hits, ['TT']*len(hits)

    def find_matching_section_of_tap(self, row): #TODO: Fix this function, different start-end section and groups. This will affect the search. RE-CHECK that search is working properly
        hits = []
        name = row['Name']
        if not BPAOutageMatcher.is_a_section_of_tap(name):
            print("Warning: Section of Tap contains invalid triggers.", name, 
                "Out Datetime:", row['Out Datetime'], "In Datetime:", row['In Datetime'])
            return [], []
        match = re.match(r'(.*) SECTION OF (.*) TAP TO (.*)', name)
        section = match.group(1)
        tap = match.group(2)
        line = match.group(3)
        for i,n in enumerate(self.line_info['Name']):
            if self._stop_after_exact_match and BPAOutageMatcher.check_section_of_tap_match_exact(n, section, tap, line):
                return [self.line_info.iloc[i]['XRefCd']], ['TS']
            if BPAOutageMatcher.check_section_of_tap_match(n, section, tap, line):
                hits.append(self.line_info.iloc[i]['XRefCd'])
        if not hits and self._outage_include_both_line_type_ends:
            new_row = row.copy()
            new_row['Name'] = tap+' TAP TO '+line
            new_row['Line Type'] = 'T'
            return self.find_matching_tap(new_row)
        return hits, ['TS']*len(hits)

    def find_matching_section(self, row): # TODO CHECK WHEN EXTENDING
        hits = []
        name = row['Name']
        if not BPAOutageMatcher.is_a_section(name):
            print("Warning: Section contains invalid triggers.", name, 
                "Out Datetime:", row['Out Datetime'], "In Datetime:", row['In Datetime'])
            return [], []
        match = re.match(r'(.*) SECTION OF (.*)', name)
        section = match.group(1)
        line = match.group(2)
        for i,n in enumerate(self.line_info['Name']):
            if self._stop_after_exact_match and BPAOutageMatcher.check_section_match_exact(n, section, line):
                return [self.line_info.iloc[i]['XRefCd']], ['S']
            if BPAOutageMatcher.check_section_match(n, section, line):
                hits.append(self.line_info.iloc[i]['XRefCd'])
        if not hits and self._outage_include_both_line_type_ends:
            new_row = row.copy()
            new_row['Name'] = line
            new_row['Line Type'] = 'L'
            return self.find_matching_lines(new_row)
        return hits, ['S']*len(hits) 

    def outage_line_name_compare(self, row):
        line_type = row['Line Type']
        # write a switch statement based on line_type - for values L, T, TT, TS
        hits = []
        hits_types = []
        if line_type == 'L':
            hits, hits_types = self.find_matching_lines(row)
        elif line_type == 'T':
            hits, hits_types = self.find_matching_tap(row)
        elif line_type == 'TT':
            hits, hits_types = self.find_matching_tap_to_tap(row)
        elif line_type == 'TS':
            hits, hits_types = self.find_matching_section_of_tap(row)
        elif line_type == 'S':
            hits, hits_types = self.find_matching_section(row)
        else:    
            hits = [np.nan]
            hits_types = [np.nan]
            print("Warning: Unknown Line Type!", row['Name'],
                "Out Datetime:", row['Out Datetime'], "In Datetime:", row['In Datetime'])
        if not hits:
            #print("Line Type: ", row['Line Type'])
            #print("Name: ", row['Name'])
            self.no_match_outages[line_type].append(row['Name'])
            return None
        row['XRefCd'] = hits
        row['XRefCd_Type'] = hits_types
        return row  

    def check_validity(self):
        """
        Check if each row's 'Line Type' is consistent with its Name.
        This is to be run BEFORE self.preprocess() to ensure data integrity.
        """
        for index, row in self.combined.iterrows():
            line_type = row['Line Type']
            if line_type == 'T':
                if not BPAOutageMatcher.is_a_tap(row['Name']):
                    print('Line Type', line_type, "Name: ", row['Name'])
                    print(f"Warning: Row {index} does not match the expected pattern for a tap")
            elif line_type == 'L':
                if not BPAOutageMatcher.is_a_line(row['Name']):
                    print(f"Warning: Row {index} does not match the expected pattern for a line")
            elif line_type == 'S':
                if not BPAOutageMatcher.is_a_section(row['Name']):
                    print(f"Warning: Row {index} does not match the expected pattern for a section")
            elif line_type == 'TT':
                if not BPAOutageMatcher.is_a_tap_to_tap(row['Name']):
                    print(f"Warning: Row {index} does not match the expected pattern for a tap to tap")
            elif line_type == 'TS':
                if not BPAOutageMatcher.is_a_section_of_tap(row['Name']):
                    print(f"Warning: Row {index} does not match the expected pattern for a section of tap")

    def preprocess(self):
        self.no_match_outages = {lt: [] for lt in self.line_types}

        self.combined_with_lines = self.combined.apply(self.outage_line_name_compare, axis=1)
        self.combined_with_lines = self.combined_with_lines.apply(pd.Series)
        self.combined_with_lines.dropna(subset=['XRefCd'], inplace=True)
        
        if self._verbose:
            counters = {lt: Counter(self.no_match_outages[lt]) for lt in self.line_types}
            for lt in self.line_types:
                print(f"Line Type: {lt}")
                for name, count in counters[lt].most_common():
                    print(f"Name: {name}, Count: {count}")
            print(tabulate(self.combined_with_lines.head(), headers="keys", tablefmt="psql"))

        self.grouped = self.combined_with_lines.groupby(['Outage ID', 'Voltage (kV)', 'Out Datetime', 'Cause', 'Responsible System']).agg({
            'In Datetime': list,
            'Original Line Name': list,
            'Line Type': list,
            'O&M District': list,
            'Length (miles)': list,
            'Duration (minutes)': list,
            'Name': list,
            'XRefCd': list,
            'XRefCd_Type': list
        })
        self.grouped = self.grouped.reset_index()
        self._remove_duplicates()

    def _remove_duplicates(self):
        duplicates = self.grouped.duplicated(subset='Outage ID', keep=False)

        if self._verbose:
            # Filter the original DataFrame to show only the duplicates
            if duplicates.any():
                print(self.grouped[duplicates])
            else:
                print("No duplicates found.")
            # Also print the number of duplicates
            print(f"Number of duplicates: {duplicates.sum()}")

        self.grouped.drop_duplicates(subset='Outage ID', inplace=True)
        duplicates = self.grouped.duplicated(subset='Outage ID', keep=False)
        if self._verbose:
            if duplicates.any():
                print(self.grouped[duplicates])
            else:
                print("No duplicates found.")
            print(tabulate(self.grouped.head(n=10), headers="keys", tablefmt="psql"))

    def print_unmatched_outage_names(self):
        """
            Print the names of unmatched lines AFTER self.preprocess() has been run
            The method takes into account known non-existing lines (not present in the provided list of BPA Transmission Lines) and does not print them
        """
        for lt in self.line_types:
            print('Missing lines of type: ', lt)
            ctr=0
            for line in self.no_match_outages[lt]: 
                if re.search(r'LAKES|SUNSET|REFINERY|MEEKER|WEYCO|GLENOMA|FOSSIL MAUPIN|'
                                r'OKANOGAN TWISP|BREW|WILLAM|EWE|119|MULLAN|PHA|CIN|TWISP OKANOGAN|'
                                r'USK PINE|WHITE CREEK|TACOMA TPWR|KELS|LAU|WILLI|7TH|CUR|BOX|OLIV|SHAW|RUNY|'
                                r'HOOP|SANTA|CHERR|DIL|USK USK|WILLAK|COLD|HAZ|RIVER ROAD|WOODL|WILLOW CREEK LANE|RATH|'
                                r'PACW WALLULA|MERW|CEDARVILLE JUNCTION TAP TO MILE HI ALTURAS|CARR|LINDE|POLE|'
                                r'MCNARY WALLULA|MILE HI ALTURAS|' # Missing lines with present taps to them
                                r'HAT ROCK TAP TO MCNARY WALLULA|DAVIS CREEK TAP TO MILE HI ALTURAS|'
                                r'KALAMA|SENECA|DELTA|HEMINGWAY|RAPIDS|SHEPHERDS', line):
                    continue
                print(line)
                ctr+=1
            print('Total unmatched: ', ctr)

    def save_csv(self, filename):
        self.grouped[['XRefCd', 'XRefCd_Type']] = self.grouped[['XRefCd', 'XRefCd_Type']].apply(lambda x: x.map(lambda y: ','.join('['+','.join(z)+']' for z in y)))
        self.grouped[['In Datetime','Original Line Name', 'Line Type', 'O&M District', 'Length (miles)', 'Duration (minutes)', 'Name']] = self.grouped[['In Datetime','Original Line Name', 'Line Type', 'O&M District', 'Length (miles)', 'Duration (minutes)', 'Name']].apply(lambda x: x.map(lambda y: ','.join(str(z) for z in y)))
        #grouped[['In Datetime']] = grouped[['In Datetime']].applymap(lambda x: ','.join(y.strftime('%Y-%m-%d %H:%M:%S') for y in x))
        self.grouped.to_csv(filename, index=False)
    
    def load_csv(self, filename='BPA_Line_Outage_Matcher/preprocessed_outages_2018_2023.csv'):
        self.grouped = pd.read_csv(Path(__file__).parent.parent /filename)
        for col in ['XRefCd', 'XRefCd_Type']:
            self.grouped[col] = self.grouped[col].apply(self.correct_and_eval_list_of_lists)
        for col in ['In Datetime','Original Line Name', 'Line Type', 'O&M District', 'Length (miles)', 'Duration (minutes)', 'Name']:
            self.grouped[col] = self.grouped[col].apply(lambda x: x.split(',') if pd.notna(x) else [float('nan')])

    def correct_and_eval_list_of_lists(self, x):
        if pd.isna(x):
            return []
        # Remove leading and trailing brackets and wrap the entire string in brackets
        try:
            x_clean = x.strip('[]')  # Removes only the outermost brackets
            corrected = '[' + ','.join(
                '[' + ','.join(f'"{item}"' for item in group.strip().split(',')) + ']'
                for group in x_clean.split('],[')
            ) + ']'
            return ast.literal_eval(corrected)
        except Exception as e:
            print(f"Error evaluating string: {x} with error {str(e)}")
            return []
        
    def time_from_last_outage(self, xrefcd, date):
        # Convert 'date' to a datetime object if it's not already
        date = pd.to_datetime(date)
        
        # Initialize variable to track the last outage date
        last_outage_date = None
        
        # Iterate through the outage dataframe
        for _, row in self.grouped.iterrows():
            out_datetime = pd.to_datetime(row['Out Datetime'])

            # Check if xrefcd is in any of the lists within the list of lists
            if any(xrefcd in group for group in row['XRefCd']):
                if out_datetime <= date:  # Only consider outages before the given date
                    if last_outage_date is None or out_datetime > last_outage_date:
                        last_outage_date = out_datetime

        # Check if there was any outage found
        if last_outage_date is None:
            return 1.0  # If no outage was ever recorded before the date, return 1 (max scaled value)

        hours_since_last_outage = (date - last_outage_date).total_seconds() / 3600
        # Cap at 2160 hours (90 days) and scale between 0 and 1
        scaled_value = min(hours_since_last_outage, 2160) / 2160
        return scaled_value
    
    def get_line_features(self, xrefcd_list):
        """
        Get line features for a list of XRefCd values using the line information dataframe in self.line_info.
        """
        line_features = {}
        filtered_line_info = self.line_info[self.line_info['XRefCd'].isin(xrefcd_list)]
        for _, row in filtered_line_info.iterrows():
            xrefcd = row['XRefCd']
            voltage = row['VoltageMeas']
            shape_length = row['SHAPE_Length']
            line_name = row['Name']

            # Determine line type using the helper methods
            line_type = [0, 0, 0]  # Corresponds to [L, T, TT]
            if self.is_a_line(line_name):
                line_type[0] = 1
            elif self.is_a_tap(line_name):
                line_type[1] = 1
            elif self.is_a_tap_to_tap(line_name):
                line_type[2] = 1
            else:
                # Throw exception existing xrefcd not matched to any type
                raise Exception(f"Existing xrefcd {xrefcd} not matched to any type")

            # Map XRefCd to features
            line_features[xrefcd] = [voltage, shape_length] + line_type

        return line_features
    
    # Initially include Wind, Ice, Weather, // NO 'RAS Initiated'
    # Later we can include: Lightning, 'Tree blown', 'Tree', 'Environmental' (LOOK Up codes)