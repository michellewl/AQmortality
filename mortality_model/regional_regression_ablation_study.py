import os
import pandas as pd
import matplotlib.pyplot as plt
from model_classes import HealthModel
from plot_functions import save_timeseries_plot
import torch
from datetime import datetime as dt

torch.manual_seed(1)

# -------------------------------------------------

window_size_universal = 7

ablation_experiments = [
    ["NO2"], ["PM10"], ["NO2", "PM10"],
    ["temperature"], ["wind_speed"], ["humidity"], ["dew_point"], ["wind_speed"], ["pressure"],
    ["temperature", "humidity", "dew_point", "wind_speed", "pressure"],
    ["NO2", "PM10", "temperature", "humidity", "dew_point", "wind_speed", "pressure"],
    ["gross_disposable_income"], ["property_income_received"], ["social_benefits_received"], ["current_taxes_on_wealth"],
    ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"]
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
    "window_size": window_size_universal # Input window size
    }

MLP_regressor_config = {
    "architecture": "MLP_regressor",
    "train_size": 0.7,
    "val_size": 0.15, # Set MLP configs
    "hidden_layer_sizes": [10],
    "batch_size": 30, # to False
    "num_epochs": 100, # if using
    "learning_rate": 0.001, # linear regressor
    "spatial_resolution": "regional",
    "temporal_resolution": "daily",
    "input_artifacts": ["laqn-regional", "met-resample", "income-regional"],
    "laqn_variables": ["NO2", "PM10"], # Set to False if excluding air pollutants
    "met_variables": ["temperature", "humidity", "dew_point", "wind_speed", "pressure"], # "wind_dir" # Set to False if excluding meteorology
    "income_variables": ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"], # Set to False if excluding income
    "ablation_features": ["PM10"], # Set to False if not analysing feature importance
    "target_shift": 1, # Time lag for target variable
    "window_size": window_size_universal # Input window size
    }

LSTM_regressor_config = {
    "architecture": "LSTM_regressor",
    "train_size": 0.7,
    "val_size": 0.15, 
    "hidden_layer_sizes": [10],
    "batch_size": 30, 
    "num_epochs": 100, 
    "learning_rate": 0.001, 
    "spatial_resolution": "regional",
    "temporal_resolution": "daily",
    "input_artifacts": ["laqn-regional", "met-resample", "income-regional"],
    "laqn_variables": ["NO2", "PM10"], # Set to False if excluding air pollutants
    "met_variables": ["temperature", "humidity", "dew_point", "wind_speed", "pressure"], # "wind_dir" # Set to False if excluding meteorology
    "income_variables": ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"], # Set to False if excluding income
    "ablation_features": ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"], # Set to False if not analysing feature importance
    "target_shift": 1, # Time lag for target variable
    "window_size": window_size_universal # Input window size
    }

configs = [linear_regressor_config, MLP_regressor_config, LSTM_regressor_config]

# ------------------------------------------------------------------------------------
for config in configs:
    print(config["architecture"])
    print("window size: ", config["window_size"])
    print("ablation: False")
    # Run the model training with full features
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
