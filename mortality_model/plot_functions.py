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


def plot_timeseries(dataframe, columns, title, units, figure_size=(8,4)):
    plt.figure(figsize=figure_size, dpi=300)
    for column in columns:
        dataframe[column].plot()
    plt.legend()
    plt.ylabel(units)
    plt.suptitle(title)
    plt.show()

def plot_on_map(data_geodataframe, map_geodataframe, column=None, map_column=None, map_cmap=None, title="Greater London", fontsize="25", figsize=(20,10), data_color=None, data_cmap=None, colorbar=False, map_colorbar=False, set_colorbar_max=False, set_map_colorbar_max=False, set_colorbar_log=False, set_map_colorbar_log=False, data_markersize=0.1, map_color="whitesmoke", map_edge_color="black", axis="off"):
    data_geodataframe.plot(column=column, ax=map_geodataframe.plot(column=map_column, figsize=figsize, color=map_color, edgecolor=map_edge_color, cmap=map_cmap), color=data_color, cmap=data_cmap, markersize=data_markersize)
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
    if map_colorbar:
        if set_colorbar_max:
            colorbar_max = set_map_colorbar_max
        else:
            colorbar_max = map_geodataframe[map_column].max()
        if set_map_colorbar_log:
            norm = colors.LogNorm(map_geodataframe[map_column].quantile(0.01), colorbar_max)
        else:
            norm = plt.Normalize(map_geodataframe[map_column].min(), colorbar_max)
        plt.colorbar(plt.cm.ScalarMappable(cmap=map_cmap, 
        norm=norm)).set_label(map_column)
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

