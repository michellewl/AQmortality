import numpy as np
import pandas as pd
import datetime as dt
import geopandas as gpd
import xarray as xr
from PyBNG import PyBNG
import shapely
from os import makedirs, path, listdir, remove
import matplotlib.pyplot as plt
import re
import requests
import zipfile as zpf
from tqdm import tqdm
from PIL import Image
from ADMS_functions import extract_dataset_time_axis, plot_timeseries, PointXYZ_to_latlon, plot_on_map, plot_in_grid_box, process_timeseries_dataset

run = "015"
folder = f"/home/users/mwlw3/ADMS-Urban/2019_hourly_met/all_regions/{run}/"
files = [path.join(folder, file) for file in listdir(folder) if path.splitext(file)[-1]==".nc"]
processed_coordinates_filepath = path.join(folder, "raw_processed_coordinates.nc")

# Processing from raw ADMS-Urban outputs to a regional netCDF files with useful attributes and latitude/longitude coordinates
processed_regions_folder = path.join(folder, "raw_processed_regions")
if files and not path.exists(processed_coordinates_filepath):
    print(f"Loading raw data for run {run} and processing the netCDF coordinates...")
    if not path.exists(processed_regions_folder):
        makedirs(processed_regions_folder)
    for i in tqdm(range(len(files))):
        region = files[i].split(".")[-2]
        region_filepath = path.join(folder, "raw_processed_regions", f"{region}.nc")
        if not path.exists(region_filepath):
            region_ds = process_timeseries_dataset(xr.open_dataset(files[i]))
            region_ds.to_netcdf(region_filepath)
        else:
            print(f"Regional file {region}.nc already exists.")
    print("Removing raw files.")
    [remove(file) for file in files]
    print("Done.")
            
# Subset for each month of the year and concatenate all regions together  
print("Processing data to create monthly files...")
files = [path.join(processed_regions_folder, file) for file in listdir(processed_regions_folder)]

processed_months_folder = path.join(folder, "raw_processed_months")
if not path.exists(processed_months_folder):
    makedirs(processed_months_folder)

for month_i in tqdm(range(12)):
    month = month_i+1
    monthly_filepath = path.join(processed_months_folder, f"month_{str(month).zfill(2)}.nc")
    if not path.exists(monthly_filepath):
        monthly_regions = []
        for files_i in range(len(files)):
            ds = xr.open_dataset(files[files_i])
            year = int(re.findall("\d\d\d\d", folder)[0])
            ds = ds.isel(time=(ds.datetime.dt.year==year))
            ds = ds.isel(time=(ds.datetime.dt.month==month))
            monthly_regions.append(ds)

        monthly_ds = xr.concat(monthly_regions, dim="space")
        monthly_ds.to_netcdf(monthly_filepath)
    else:
        print(f"Monthly file month_{str(month).zfill(2)}.nc already exists.")
print("Removing regional files.")
[remove(file) for file in files]
print("Done.")
#     # loop through processed region files... 
#     print("Concatenating netCDF files...")
#     new_ds = xr.concat([xr.open_dataset(path.join(processed_regions_folder, listdir(processed_regions_folder)[i])) for i in tqdm(range(len(listdir(processed_regions_folder)[0:10])))], "space")
   
#     print("Saving new netCDF file...")
#     new_ds.to_netcdf(processed_coordinates_filepath)
#     print(f"Saved to {processed_coordinates_filepath}")

# elif path.exists(processed_coordinates_filepath):
#     new_ds = xr.open_dataset(processed_coordinates_filepath)
#     print(f"Loaded the processed coordinate data for run {run}.")

