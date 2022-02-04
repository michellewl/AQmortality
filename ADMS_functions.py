import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from PyBNG import PyBNG
import shapely
from os import makedirs, path, listdir, remove
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import requests
import zipfile as zpf
from tqdm import tqdm
from PIL import Image

def metline_to_datetime(i):
    numbers = str(i).replace("b", "").replace("'", "").split("_")
    return np.datetime64(f"{numbers[0]}")+ np.timedelta64(int(numbers[1])-1, "D") + np.timedelta64(int(numbers[2]), "h")

def extract_dataset_time_axis(ds):
    datetime_index = []
    for i in ds.Met_Line.values:
        datetime_index.append(metline_to_datetime(i))
    return datetime_index

def plot_timeseries(dataframe, columns, title, units, figure_size=(8,4)):
    plt.figure(figsize=figure_size, dpi=300)
    for column in columns:
        dataframe[column].plot()
    plt.legend()
    plt.ylabel(units)
    plt.suptitle(title)
    plt.show()

def PointXYZ_to_latlon(PointXs, PointYs):
    df = pd.DataFrame()
    for X, Y in zip(PointXs.astype(int), PointYs.astype(int)):
        latlon = PyBNG(easting=X, northing=Y).get_latlon()
        df = df.append([latlon])
    df.columns = ["latitude", "longitude"]
    df.reset_index(drop=True, inplace=True)
    return df

def plot_on_map(data_geodataframe, map_geodataframe, column=None, title="Greater London", fontsize="25", figsize=(20,10), data_color=None, data_cmap=None, colorbar=False, set_colorbar_max=False, set_colorbar_log=False, data_markersize=0.1, map_color="whitesmoke", map_edge_color="black", axis="off"):
    data_geodataframe.plot(column=column, ax=map_geodataframe.plot(figsize=figsize, color=map_color, edgecolor=map_edge_color), color=data_color, cmap=data_cmap, markersize=data_markersize)
    if colorbar:
        if set_colorbar_max:
            colorbar_max = set_colorbar_max
        else:
            colorbar_max = data_geodataframe[column].max()
        if set_colorbar_log:
            norm = colors.LogNorm(data_geodataframe[column].quantile(0.01), colorbar_max)
        else:
            norm = plt.Normalize(data_geodataframe[column].min(), colorbar_max)
        plt.colorbar(plt.cm.ScalarMappable(cmap=data_cmap, 
        norm=norm)).set_label(column)
    plt.suptitle(title, fontsize=fontsize)
    plt.axis(axis)
    plt.show()
    
def plot_in_grid_box(geodataframe, column, title, figsize=(10,5), fontsize=15, markersize=0.1, cmap="plasma_r", colorbar=True, colorbar_label=None, colorbar_scale=False, edgecolor=None):
    geodataframe.plot(column=column, figsize=figsize, markersize=markersize, cmap=cmap, edgecolor=edgecolor)
    colorbar_max = geodataframe[column].max()
    if colorbar_scale == "log":
        norm = colors.LogNorm(geodataframe[column].quantile(0.01), colorbar_max)
    else:
        norm = plt.Normalize(geodataframe[column].min(), colorbar_max)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), label=colorbar_label)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.suptitle(title, fontsize=fontsize)
    plt.show()

def process_timeseries_dataset(ds):
    data_variables = {
        "NOx": (["time", "space"], ds.Dataset1.values.squeeze(), ds.Dataset1.attrs),
        "NO2": (["time", "space"], ds.Dataset2.values.squeeze(), ds.Dataset2.attrs),
        "PM10": (["time", "space"], ds.Dataset3.values.squeeze(), ds.Dataset3.attrs),
        "PM2.5": (["time", "space"], ds.Dataset4.values.squeeze(), ds.Dataset4.attrs),
        "CO2": (["time", "space"], ds.Dataset5.values.squeeze(), ds.Dataset5.attrs),
        "O3": (["time", "space"], ds.Dataset6.values.squeeze(), ds.Dataset6.attrs),
        "wind_speed_at_10m": (["time"], ds.Met_UAt10m.values.squeeze(), ds.Met_UAt10m.attrs),
        "wind_direction": (["time"], ds.Met_PHI.values.squeeze(), ds.Met_PHI.attrs)
                     }

    coords = {"datetime": (["time"], np.array(extract_dataset_time_axis(ds))),
         "latitude": (["space"], PointXYZ_to_latlon(ds.PointX_XYZ.values, ds.PointY_XYZ.values).latitude.values),
         "longitude": (["space"], PointXYZ_to_latlon(ds.PointX_XYZ.values, ds.PointY_XYZ.values).longitude.values)}

    attrs = ds.attrs

    return xr.Dataset(data_vars=data_variables, coords=coords, attrs=attrs)
    
def process_PG_dataset(ds):
    data_variables = {
        ds.Dataset1.Pollutant_Name: (["PG_class", "space"], ds.Dataset1.values.reshape((ds.dims["nMetLines"],-1)), ds.Dataset1.attrs),
        ds.Dataset2.Pollutant_Name: (["PG_class", "space"], ds.Dataset2.values.reshape((ds.dims["nMetLines"],-1)), ds.Dataset2.attrs),
        ds.Dataset3.Pollutant_Name: (["PG_class", "space"], ds.Dataset3.values.reshape((ds.dims["nMetLines"],-1)), ds.Dataset3.attrs),
        ds.Dataset4.Pollutant_Name: (["PG_class", "space"], ds.Dataset4.values.reshape((ds.dims["nMetLines"],-1)), ds.Dataset4.attrs),
        ds.Dataset5.Pollutant_Name: (["PG_class", "space"], ds.Dataset5.values.reshape((ds.dims["nMetLines"],-1)), ds.Dataset5.attrs),
        "wind_speed_at_10m": (["PG_class"], ds.Met_UAt10m.values, ds.Met_UAt10m.attrs),
        "wind_direction": (["PG_class"], ds.Met_PHI.values, ds.Met_PHI.attrs)
                     }

    coords = {"Pasquill-Gifford": (["PG_class"], np.array(PG_index)),
         "latitude": (["space"], PointXYZ_to_latlon(ds.PointX_XYZ.values, ds.PointY_XYZ.values).latitude.values),
         "longitude": (["space"], PointXYZ_to_latlon(ds.PointX_XYZ.values, ds.PointY_XYZ.values).longitude.values)}

    attrs = ds.attrs

    return xr.Dataset(data_vars=data_variables, coords=coords, attrs=attrs)

