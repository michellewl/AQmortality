import numpy as np
import pandas as pd
import datetime as dt
import geopandas as gpd
import xarray as xr
from PyBNG import PyBNG
import shapely
from os import makedirs, path, listdir, remove
import matplotlib.pyplot as plt
import requests
import zipfile as zpf
from tqdm import tqdm
from PIL import Image
from ADMS_functions import extract_dataset_time_axis, plot_timeseries, PointXYZ_to_latlon, plot_on_map, plot_in_grid_box, process_timeseries_dataset

run = "015"
folder = f"/home/users/mwlw3/ADMS-Urban/2019_hourly_met/all_regions/{run}/"
files = [path.join(folder, file) for file in listdir(folder) if path.splitext(file)[-1]==".nc"]
processed_coordinates_filepath = path.join(folder, "raw_processed_coordinates.nc")

# Processing from raw ADMS-Urban outputs to a netCDF file with useful attributes and latitude/longitude coordinates

if not path.exists(processed_coordinates_filepath):
    print(f"Loading raw data for run {run} and processing the netCDF coordinates...")
    processed_regions_folder = path.join(folder, "raw_processed_regions")
    if not path.exists(processed_regions_folder):
        makedirs(processed_regions_folder)
    for i in tqdm(range(len(files))):
        region = files[i].split(".")[-2]
        region_filepath = path.join(folder, "raw_processed_regions", f"{region}.nc")
        if not path.exists(region_filepath):
            region_ds = process_timeseries_dataset(xr.open_dataset(files[i]))
            region_ds.to_netcdf(region_filepath)
    # loop through processed region files... 
    print("Concatenating netCDF files...")
    new_ds = xr.concat([xr.open_dataset(path.join(processed_regions_folder, listdir(processed_regions_folder)[i])) for i in tqdm(range(len(listdir(processed_regions_folder)[0:10])))], "space")
   
    print("Saving new netCDF file...")
    new_ds.to_netcdf(processed_coordinates_filepath)
    print(f"Saved to {processed_coordinates_filepath}")

elif path.exists(processed_coordinates_filepath):
    new_ds = xr.open_dataset(processed_coordinates_filepath)
    print(f"Loaded the processed coordinate data for run {run}.")

