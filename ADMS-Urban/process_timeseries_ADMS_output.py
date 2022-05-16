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

if not path.exists(processed_coordinates_filepath):
    print(f"Loading raw data for run {run} and processing the netCDF coordinates...")
    new_ds = xr.concat([process_timeseries_dataset(xr.open_dataset(files[i])) for i in tqdm(range(len(files)))], "space")
    new_ds.to_netcdf(processed_coordinates_filepath)
elif path.exists(processed_coordinates_filepath):
        new_ds = xr.open_dataset(processed_coordinates_filepath)
        print(f"Loaded the processed coordinate data for run {run}.")