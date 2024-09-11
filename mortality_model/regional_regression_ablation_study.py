import os
import pandas as pd
import matplotlib.pyplot as plt
from model_classes import HealthModel
from plot_functions import save_timeseries_plot
import torch
from datetime import datetime as dt
import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument("-w", "--window_size", type=int, default=7)
parser.add_argument("-rs", "--random_seed", type=int, default=1)

parser.add_argument('-e', '--MLP_num_epochs', type=int, default=500) 
parser.add_argument("-hl", "--MLP_hidden_layer_size", type=int, default=100)
parser.add_argument("-nl", "--MLP_num_layers", type=int, default=4)
parser.add_argument("-b", "--MLP_batch_size", type=int, default=64)
parser.add_argument("-lr", "--MLP_learning_rate", type=float, default=0.01)

parser.add_argument('-e', '--LSTM_num_epochs', type=int, default=500) 
parser.add_argument("-hl", "--LSTM_hidden_layer_size", type=int, default=100)
parser.add_argument("-nl", "--LSTM_num_layers", type=int, default=1)
parser.add_argument("-b", "--LSTM_batch_size", type=int, default=64)
parser.add_argument("-lr", "--LSTM_learning_rate", type=float, default=0.001)

args = parser.parse_args()

torch.manual_seed(args.random_seed)

# -------------------------------------------------

ablation_experiments = [
    ["NO2"], ["PM10"], 
    ["NO2", "PM10"], # grouped pollution
    ["temperature"], ["wind_speed"], ["humidity"], ["dew_point"], ["wind_speed"], ["pressure"],
    ["temperature", "humidity", "dew_point", "wind_speed", "pressure"], # grouped meteorology
    ["NO2", "PM10", "temperature", "humidity", "dew_point", "wind_speed", "pressure"], # grouped environmental
    ["gross_disposable_income"], ["property_income_received"], ["social_benefits_received"], ["current_taxes_on_wealth"],
    ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"] # group socioeconomics
]

# -------------------------------------------------

linear_regressor_config = {
    "architecture": "linear_regressor",
    "train_size": 0.8,
    "val_size": False, # Set MLP 
    "hidden_layer_sizes": False, # configs
    "batch_size": False, # to False
    "num_epochs": False, # if using
    "learning_rate": False, # linear regressor
    "spatial_resolution": "regional",
    "temporal_resolution": "daily",
    "input_artifacts": ["laqn-regional", "met-resample", "income-regional"],
    "laqn_variables": ["NO2", "PM10"], # Set to False if excluding air pollutants
    "met_variables": ["temperature", "humidity", "dew_point", "wind_speed", "pressure"], # "wind_dir"# Set to False if excluding meteorology
    "income_variables": ["gross_disposable_income", "property_income_received", #"property_income_paid", 
                         "social_benefits_received", "current_taxes_on_wealth"], # Set to False if excluding income
    "ablation_features": False, #["temperature"] # Set to False if not analysing feature importance
    "target_shift": 1, # Time lag end point for target variable
    "window_size": args.window_size # Input window size
    }

MLP_regressor_config = {
    "architecture": "MLP_regressor",
    "train_size": 0.7,
    "val_size": 0.15, 
    "hidden_layer_size": args.MLP_hidden_layer_size, 
    "num_layers": args.MLP_num_layers,
    "batch_size": args.MLP_batch_size, 
    "num_epochs": args.MLP_num_epochs, 
    "learning_rate": args.MLP_learning_rate, 
    "spatial_resolution": "regional",
    "temporal_resolution": "daily",
    "input_artifacts": ["laqn-regional", "met-resample", "income-regional"],
    "laqn_variables": ["NO2", "PM10"], # Set to False if excluding air pollutants
    "met_variables": ["temperature", "humidity", "dew_point", "wind_speed", "pressure"], # "wind_dir" # Set to False if excluding meteorology
    "income_variables": ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"], # Set to False if excluding income
    "ablation_features": ["PM10"], # Set to False if not analysing feature importance
    "target_shift": 1, # Time lag for target variable
    "window_size": args.window_size # Input window size
    }

LSTM_regressor_config = {
    "architecture": "LSTM_regressor",
    "train_size": 0.7,
    "val_size": 0.15, 
    "hidden_layer_size": args.LSTM_hidden_layer_size,
    "num_layers": args.LSTM_num_layers, 
    "batch_size": args.LSTM_batch_size, 
    "num_epochs": args.LSTM_num_epochs, 
    "learning_rate": args.LSTM_learning_rate, 
    "spatial_resolution": "regional",
    "temporal_resolution": "daily",
    "input_artifacts": ["laqn-regional", "met-resample", "income-regional"],
    "laqn_variables": ["NO2", "PM10"], # Set to False if excluding air pollutants
    "met_variables": ["temperature", "humidity", "dew_point", "wind_speed", "pressure"], # "wind_dir" # Set to False if excluding meteorology
    "income_variables": ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"], # Set to False if excluding income
    "ablation_features": ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"], # Set to False if not analysing feature importance
    "target_shift": 1, # Time lag for target variable
    "window_size": args.window_size # Input window size
    }

configs = [linear_regressor_config, MLP_regressor_config, LSTM_regressor_config]

# ------------------------------------------------------------------------------------
for config in configs:
    print(config["architecture"])
    print("window size: ", config["window_size"])
    print("ablation: False")
    print("num_epochs", config["num_epochs"])
    print("hidden_layer_sizes", config["hidden_layer_sizes"])
    print("batch_size", config["batch_size"])
    print("learning_rate", config["learning_rate"])
    
    # Run the model training with full features first
    config["ablation_features"] = False
    model = HealthModel(config)
    inputs, targets, datetime = model.preprocess_and_log()
    model.train_and_log()
    data_dict = model.test_and_log()
    save_timeseries_plot(config, data_dict)

    for ablation_exp in ablation_experiments:
        print(ablation_exp)
        config["ablation_features"] = ablation_exp
        model = HealthModel(config)
        inputs, targets, datetime = model.preprocess_and_log()
        data_dict = model.test_and_log()
        save_timeseries_plot(config, data_dict)

# ------------------------------------------------------------------------------------
