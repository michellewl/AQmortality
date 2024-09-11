from model_classes import HealthModel
from plot_functions import save_timeseries_plot
import torch
import argparse
import wandb

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument("-sw", "--sweep", action="store_true", default=False)

parser.add_argument("-w", "--window_size", type=int, default=7)
parser.add_argument("-rs", "--random_seed", type=int, default=1)
parser.add_argument("-m", "--model", type=str, default="LSTM")

parser.add_argument('-e', '--num_epochs', type=int, default=500) 
parser.add_argument("-hl", "--hidden_layer_size", type=int, default=100)
parser.add_argument("-nl", "--num_layers", type=int, default=4)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)

args = parser.parse_args()

torch.manual_seed(args.random_seed)

# -------------------------------------------------

config = {
    "train_size": 0.7,
    "val_size": 0.15, 
    "hidden_layer_size": args.hidden_layer_size,
    "num_layers": args.num_layers,
    "batch_size": args.batch_size, 
    "num_epochs": args.num_epochs, 
    "learning_rate": args.learning_rate, 
    "spatial_resolution": "regional",
    "temporal_resolution": "daily",
    "input_artifacts": ["laqn-regional", "met-resample", "income-regional"],
    "laqn_variables": ["NO2", "PM10"], # Set to False if excluding air pollutants
    "met_variables": ["temperature", "humidity", "dew_point", "wind_speed", "pressure"], # "wind_dir" # Set to False if excluding meteorology
    "income_variables": ["gross_disposable_income", "property_income_received", "social_benefits_received", "current_taxes_on_wealth"], # Set to False if excluding income
    "ablation_features": False, # Set to False if not analysing feature importance
    "target_shift": 1, # Time lag for target variable
    "window_size": args.window_size # Input window size
    }

if "MLP" in args.model:
    config["architecture"] = "MLP_regressor" 
elif "LSTM" in args.model:
    config["architecture"] = "LSTM_regressor"

# ------------------------------------------------------------------------------------

def main():
    wandb.init(project="AQmortality")

    config.update(wandb.config)

    print(config["architecture"])
    print("window size: ", config["window_size"])
    print("random_seed: ", args.random_seed)
    print("num_epochs", config["num_epochs"])
    print("hidden_layer_size", config["hidden_layer_size"])
    print("num_layers", config["num_layers"])
    print("batch_size", config["batch_size"])
    print("learning_rate", config["learning_rate"])

    model = HealthModel(config)
    inputs, targets, datetime = model.preprocess_and_log()
    model.train_and_log()
    data_dict = model.test_and_log()
    save_timeseries_plot(config, data_dict)

    wandb.finish()

# ------------------------------------------------------------------------------------

sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "mse_train"},
    "parameters": {
        "learning_rate": {"values": [0.0001, 0.001, 0.01, 0.1]},
        "num_layers": {"values": [1, 2, 3, 4]}
    }
}

# ------------------------------------------------------------------------------------
if args.sweep:
    sweep_id = wandb.sweep(sweep=sweep_config, project="AQmortality")
    wandb.agent(sweep_id, function=main, count=10)
else:
    main()