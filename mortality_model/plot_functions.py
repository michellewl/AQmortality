import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime as dt
from os import makedirs, path, listdir, remove
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import requests
import zipfile as zpf
from tqdm import tqdm


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

def save_timeseries_plot(config, data_dict):
    if not config["ablation_features"]:
        filename = f"ablation-0.pdf"
    else:
        filename = f"ablation-{','.join(config['ablation_features'])}.pdf"
    
    directory = f"figures/window_{config['window_size']}/{config['architecture']}"
    

    # Create the directory, do not raise an error if it already exists
    makedirs(directory, exist_ok=True)

    # Set up the dataframe to plot data
    subsets = ["train", "val", "test"] if config["val_size"] else ["train", "test"]
    df_list = []
    for subset in subsets:
        subset_df = pd.DataFrame(index=pd.DatetimeIndex(data_dict[f"{subset}_dates"]), data={"observed":data_dict[f"y_{subset}"].flatten(), "predicted":data_dict[f"y_{subset}_predict"].flatten()})
        df_list.append(subset_df)
    df = pd.concat(df_list)

    # Create the timeseries plot
    plt.figure(figsize=(8,5), dpi=150)
    df["observed"].plot()
    df["predicted"].plot()
    plt.axvline(data_dict["train_dates"].max(), color="grey")
    plt.axvline(data_dict["val_dates"].max(), color="grey") if config["val_size"] else None
    plt.legend()
    plt.ylabel("deaths per 100,000")
    plt.xlim(dt(year=1995, month=1, day=1), dt(year=2020, month=1, day=1))
    # regressor_title = config["architecture"].replace("_", " ")
    # plt.suptitle(f"Mortality predictions by {regressor_title} {plot_title_model}")

    # Save the plot
    file_path = path.join(directory, filename)
    plt.savefig(file_path)

    plt.close()