# Imports

import numpy as np
import requests
import pandas as pd
from os import makedirs, path, listdir
from tqdm import tqdm
import zipfile as zpf
import matplotlib.pyplot as plt
import xlrd
from openpyxl import load_workbook

# London Air Quality Network data class

class LAQNData():
    def __init__(self, url, home_folder, species, start_date, end_date):
        self.url = url
        self.home_folder = home_folder
        self.species = species
        self.start_date = start_date
        self.end_date = end_date
        self.filename = f"{self.species}_hourly_{self.start_date}_{self.end_date}.csv"
        self.filepath = path.join(self.home_folder, self.filename)
        
        if not path.exists(self.home_folder):
            makedirs(self.home_folder)
        
        london_sites = requests.get(self.url)
        self.sites_df = pd.DataFrame(london_sites.json()['Sites']['Site'])
        self.site_codes = self.sites_df["@SiteCode"].tolist()

    def download(self, verbose=True):
        laqn_df = pd.DataFrame()
        
        if verbose:
            progress_bar = tqdm(self.site_codes)
        else:
            progress_bar = self.site_codes
            
        for site_code in progress_bar:
            if verbose:
                progress_bar.set_description(f'Working on site {site_code}')
            url_species = f"http://api.erg.kcl.ac.uk/AirQuality/Data/SiteSpecies/SiteCode={site_code}/SpeciesCode={self.species}/StartDate={self.start_date}/EndDate={self.end_date}/csv"
            cur_df = pd.read_csv(url_species)
            cur_df.columns = ["date", site_code]
            cur_df.set_index("date", drop=True, inplace=True)

            try:
                if laqn_df.empty:
                    laqn_df = cur_df.copy()
                else:
                    laqn_df = laqn_df.join(cur_df.copy(), how="outer")

            except ValueError:  # Trying to join with duplicate column names
                rename_dict = {}
                for x in list(set(cur_df.columns).intersection(laqn_df.columns)):
                    rename_dict.update({x: f"{x}_"})
                    print(f"Renamed duplicated column:\n{rename_dict}")
                laqn_df.rename(mapper=rename_dict, axis="columns", inplace=True)
                if laqn_df.empty:
                    laqn_df = cur_df.copy()
                else:
                    laqn_df = laqn_df.join(cur_df.copy(), how="outer")
                if verbose:
                    print(f"Joined.")

            except KeyError:  # Trying to join along indexes that don't match
                print(f"Troubleshooting {site_code}...")
                cur_df.index = cur_df.index + ":00"
                if laqn_df.empty:
                    laqn_df = cur_df.copy()
                else:
                    laqn_df = laqn_df.join(cur_df.copy(), how="outer")
                print(f"{site_code} joined.")

        print("Data download complete. Removing sites with 0 data...")
        laqn_df.dropna(axis="columns", how="all", inplace=True)
        laqn_df.to_csv(path.join(self.home_folder, self.filename))
        print("Data saved.")

    def read_csv(self, verbose=True, index_col="date", parse_dates=True):
        if verbose:
            print(f"Reading {self.filename}...")
        return pd.read_csv(self.filepath, index_col=index_col, parse_dates=parse_dates)
    
    def resample_time(self, df, key, quantile_step, verbose=True):
        if key == "D":
            keyword = "daily"
        if key == "W":
            keyword = "weekly"

        save_folder = path.join(self.home_folder, keyword)
        if not path.exists(save_folder):
            makedirs(save_folder)

        aggregation = np.round(np.arange(0, 1 + quantile_step, quantile_step), 2).tolist()

        for method in aggregation:
            aggregated_df = df.copy().resample(key).quantile(method)
            method = f"{int(method * 100)}th_quantile"
            aggregated_df.to_csv(path.join(save_folder, f"{self.species}_{keyword}_{method}.csv"), index=True)
            if verbose:
                print(f"Dataframe shape {aggregated_df.shape}")
        if verbose:
            print("Done.")
            
# Office for National Statistics health data class

# Data class definition

class HealthData():
    def __init__(self, home_folder, url=False, filename=False):
        self.home_folder = home_folder
        if url:
            self.url = url
            self.filename = path.basename(self.url)
        elif filename:
            self.filename = filename
        _, self.extension = path.splitext(self.filename)
        self.filepath = path.join(self.home_folder, self.filename)

        if not path.exists(self.home_folder):
            makedirs(self.home_folder)

    def download(self, verbose=True):
        request = requests.get(self.url)
        file = open(self.filepath, 'wb')
        file.write(request.content)
        file.close()
        if verbose:
            print(f"Saved to {self.filename}")
        if self.extension == ".zip":
            self.zipfiles = zpf.ZipFile(self.filepath).namelist()
            if verbose:
                print("Contains zip files:")
                [print(f"[{i}] {self.zipfiles[i]}") for i in range(len(self.zipfiles))]
        elif self.extension == ".xls":
            workbook = xlrd.open_workbook(self.filepath)
            self.sheets = workbook.sheet_names()
            if verbose:
                print(f"Contains xls sheets: {self.sheets}")
        elif self.extension == ".xlsx":
            workbook = load_workbook(self.filepath)
            self.sheets = workbook.sheetnames
            if verbose:
                print(f"Contains xlsx sheets: {self.sheets}")
                
    def unzip(self, file="all", verbose=True):
        with zpf.ZipFile(self.filepath, 'r') as zip_ref:
            if file =="all":                                       # Extract all zipped files.
                zip_ref.extractall(self.home_folder)
            else:                                                  # Extract one specified file,
                zip_ref.extract(file, self.home_folder)            # then
                self.filename = path.basename(file)                # reset the file name, path and extension info.
                _, self.extension = path.splitext(self.filename)
                self.filepath = path.join(self.home_folder, self.filename)
        if verbose:
            print(f"Unzipped {file}.")

    def read_csv(self, verbose=True, index_col="date", parse_dates=True):
        if verbose:
            print(f"Reading {self.filename}...")
        return pd.read_csv(self.filepath, index_col=index_col, parse_dates=parse_dates)
    
    def read_xls(self, sheet_name, verbose=True):
        if verbose:
            print(f"Reading {self.filename}...")
        if self.extension == ".xls":
            return pd.read_excel(self.filepath, sheet_name)
        elif self.extension == ".xlsx":
            workbook = load_workbook(self.filepath)
            worksheet = workbook[sheet_name]
            return pd.DataFrame(worksheet.values)