# Imports for classes
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from os import path, listdir, environ
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from copy import deepcopy

# Health model class for linear or neural network regression models.

class HealthModel():
    def __init__(self, config):
        self.config = config
        self.architecture = config["architecture"]
        self.train_size = config["train_size"]
        self.val_size = config["val_size"]
        if config["hidden_layer_sizes"]:
            self.hidden_layer_sizes = config["hidden_layer_sizes"]
        #     if config["met_variables"]:
        #         self.hidden_layer_sizes = [len(config["input_artifacts"])+len(config["met_variables"])-1] +config["hidden_layer_sizes"]
        #     else:
        #         self.hidden_layer_sizes = [len(config["input_artifacts"])] +config["hidden_layer_sizes"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.spatial_resolution = config["spatial_resolution"]
        self.temporal_resolution = config["temporal_resolution"]
        self.input_artifacts = config["input_artifacts"]
        self.laqn_variables = config["laqn_variables"]
        self.met_variables = config["met_variables"]
        self.income_variables = config["income_variables"]
        self.target_shift = config["target_shift"]
        self.ablation_features = config["ablation_features"]
        self.window_size = config["window_size"]

    def preprocess_and_log(self):
        with wandb.init(project="AQmortality", job_type="split-normalise-data", mode="online") as run:
            df = pd.DataFrame()
            # use dataset artifacts
            for artifact in self.input_artifacts:
                print(artifact)
                data_artifact = run.use_artifact(f"{artifact}:latest")
                data_folder = data_artifact.download()
                if artifact == "met-resample":
                    dfs = []  # List to hold individual DataFrames for each variable
                    for variable in self.met_variables:
                        # file = f"{variable}.npz"
                        # data = np.load(path.join(data_folder, file), allow_pickle=True)
                        # if df.empty:
                        #     df = pd.DataFrame(index=pd.DatetimeIndex(data["x"]), data=data["y"], columns=[variable])
                        # else:
                        #     df = df.join(pd.DataFrame(index=pd.DatetimeIndex(data["x"]), data=data["y"], columns=[variable]))

                        filepath = path.join(data_folder, f"{variable}.npz")
                        data = np.load(filepath, allow_pickle=True)
                        variable_data = {}  # Dictionary to hold data for each statistic

                        # Extract each statistic (mean, min, max) for the current variable
                        for stat in ['mean', 'min', 'max']:
                            column_name = f"{variable}_{stat}"  # Construct column name
                            variable_data[stat] = pd.Series(data[stat], index=data['x'], name=column_name)

                        # Concatenate the statistics for the current variable into a single DataFrame
                        variable_df = pd.concat(variable_data.values(), axis=1)

                        # Append the DataFrame for the current variable to the list
                        dfs.append(variable_df)

                    # Concatenate all DataFrames in the list along the columns axis
                    if df.empty:
                        df = pd.concat(dfs, axis=1).copy()
                    else:
                        df = df.join(pd.concat(dfs, axis=1))

                elif artifact == "laqn-regional":
                    for variable in self.laqn_variables:
                        for file in listdir(data_folder):
                            if variable in file:
                                column = file.replace(".npz", "")
                                filepath = path.join(data_folder, file)
                                data = np.load(filepath, allow_pickle=True)
                                # print(data["y"].shape)
                                if df.empty:
                                    df = pd.DataFrame(index=pd.DatetimeIndex(data["x"]), data=data["y"], columns=[column])
                                else:
                                    df = df.join(pd.DataFrame(index=pd.DatetimeIndex(data["x"]), data=data["y"], columns=[column]))

                elif artifact == "income-regional":
                    dfs = []
                    for variable in self.income_variables:
                        for file in listdir(data_folder):
                            if variable in file:
                                data = np.load(path.join(data_folder, file), allow_pickle=True)
                                feature_df = pd.DataFrame(index=pd.DatetimeIndex(data["x"]), data=data["y"], columns=[file.replace(".npz", "")])
                                dfs.append(feature_df)
                    # Concatenate all DataFrames in the list along the columns axis
                    if df.empty:
                        df = pd.concat(dfs, axis=1).copy()
                    else:
                        df = df.join(pd.concat(dfs, axis=1))
                else:
                    print(f"input_artifact {artifact} not recognised.")
            input_columns = df.columns
            print("input columns: ", input_columns)

            df = df.dropna(axis=0) # Drop NaN values before windowing

            # Ablation study
            if self.ablation_features:
                # Identify columns
                columns = []
                for ablation_feature in self.ablation_features:
                    for col in df.columns:
                        if ablation_feature in col:
                            columns.append(col)
                    # columns = [col for col in df.columns if ablation_feature in col]
                print(f"Ablation - scrambled columns: {columns}")

                for column in columns:
                    # Find mean and standard deviation of feature
                    mean = df[column].mean()
                    std = df[column].std()
                    # Replace column with random values from normal distribution
                    df[column] = np.random.normal(mean, std, df[column].shape[0])
            
            # Load mortality target data
            target_artifact = run.use_artifact("mortality-scaled:latest")
            target_folder = target_artifact.download()
            data = np.load(path.join(target_folder, "deaths.npz"), allow_pickle=True)
            mortality_data = pd.DataFrame(index=pd.DatetimeIndex(data["x"]), data=data["y"]*100000, columns=["deaths"])
            df = df.join(mortality_data)

            # make new train, validation and test artifacts for regional scale data
            if self.val_size:
                index = {"train": df.index[:int(len(df.index)*self.train_size)],
                            "val": df.index[int(len(df.index)*self.train_size):int(len(df.index)*(self.train_size+self.val_size))],
                        "test": df.index[int(len(df.index)*(self.train_size+self.val_size)):]}
                subsets = ["train", "val", "test"]
            else:
                index = {"train": df.index[:int(len(df.index)*self.train_size)],
                        "test": df.index[int(len(df.index)*self.train_size):]}
                subsets = ["train", "test"]

            scaler = MinMaxScaler()
            x_scaler = scaler.fit(df.loc[index["train"]].drop("deaths", axis=1)) # Fit the scaler only to the training set distribution
            input_list = []
            targets_list = []
            datetime_list = []
            for subset in subsets:
                print("\n", subset)
                inputs = x_scaler.transform(df.loc[index[subset]].drop("deaths", axis=1)) # Apply the scaler to each subset
                targets = df.loc[index[subset]]["deaths"].values
                datetime = df.loc[index[subset]].index
                print("inputs", inputs.shape)
                print("targets", targets.shape)
                print("datetime index", datetime.shape)
                input_df = pd.DataFrame(index=pd.DatetimeIndex(datetime), data=inputs, columns=input_columns)
                target_df = pd.DataFrame(index=pd.DatetimeIndex(datetime), data=targets, columns=["deaths"])

                # Apply windowing function to input features and offset by time lag
                print(f"Processing time lagged ({self.target_shift}) input windows (length {self.window_size})")
                datetime, inputs = create_windowed_features(input_df, self.target_shift, self.window_size)

                target_df = target_df.loc[datetime] # Match target dates with input dates 
                targets = target_df.values

                input_list.append(inputs)
                targets_list.append(targets)
                datetime_list.append(datetime)
                print("inputs (before flattening)", inputs.shape)
                print("targets (before flattening)", targets.shape)
                print("datetime index", datetime.shape)
            
                subset_data = wandb.Artifact(
                            f"xy_{subset}", type="dataset",
                            description=f"Input features (normalised) and targets for {subset}ing set.",
                            metadata={"input_shape":inputs.shape,
                                     "target_shape":targets.shape,
                                     "target_shift":self.target_shift,
                                      "spatial_resolution": self.spatial_resolution,
                                      "temporal_resolution": self.temporal_resolution,
                                      "input_artifacts": self.input_artifacts,
                                      "met_variables": self.met_variables,
                                      "config": self.config})
                with subset_data.new_file(subset + ".npz", mode="wb") as file:
                    np.savez(file, x=inputs, y=targets, z=datetime)
                run.log_artifact(subset_data)
        return input_list, targets_list, datetime_list
                
#     def read_data(self, artifact):
#         with wandb.init(project="AQmortality", job_type="read-data") as run:
#             data_artifact = run.use_artifact(f"{artifact}:latest")
#             data_folder = data_artifact.download()
#             file = artifact.replace("xy_", "") + ".npz"
#             data = np.load(path.join(data_folder, file), allow_pickle=True)
#         return data["x"], data["y"]
    
    def train_and_log(self):
        model_type = self.architecture.replace("_", "-")
        print(model_type)
        data_dict = {}
        with wandb.init(project="AQmortality", job_type="train-regional-model", config=self.config) as run:
            if self.val_size:
                subsets = ["train", "val"]
            else:
                subsets = ["train"]
            for subset in subsets:
                data_artifact = run.use_artifact(f"xy_{subset}:latest")
                data_folder = data_artifact.download()
                file = f"{subset}.npz"
                data = np.load(path.join(data_folder, file), allow_pickle=True)
                data_dict.update({"x_"+subset: data["x"], "y_"+subset: data["y"]})
            if model_type == "linear-regressor":
                x_train = data_dict["x_train"].reshape(data_dict["x_train"].shape[0], -1)
                y_train = data_dict["y_train"]
                regressor = LinearRegression().fit(x_train, y_train)
                data_dict.update({"y_predict": regressor.predict(x_train)})
                wandb.log({"r_squared_train": r2_score(data_dict["y_train"], data_dict["y_predict"]),
                       "mse_train": mean_squared_error(data_dict["y_train"], data_dict["y_predict"]),
                       "mape_train": mape_score(data_dict["y_train"], data_dict["y_predict"]),
                       "rmse_train": rmse_error(data_dict["y_train"], data_dict["y_predict"]),
                       "smape_train": smape_error(data_dict["y_train"], data_dict["y_predict"])
                      })
            elif model_type == "MLP-regressor":
                print("x_train shape", data_dict["x_train"].shape)
                if len(data_dict["x_train"].shape) == 2:
                    hidden_layer_sizes = [data_dict["x_train"].shape[1]] + self.hidden_layer_sizes
                elif len(data_dict["x_train"].shape) == 3:
                    hidden_layer_sizes = [data_dict["x_train"].shape[1] * data_dict["x_train"].shape[2]] + self.hidden_layer_sizes
                else:
                    raise ValueError(f"Error in calculating network layer sizes. x_train has shape {data_dict['x_train'].shape}")
                print("hidden layer sizes", hidden_layer_sizes)
                regressor, epoch = MLPRegression(hidden_layer_sizes).fit(data_dict["x_train"], data_dict["y_train"], 
                                                                              data_dict["x_val"], data_dict["y_val"], 
                                                                              self.batch_size, self.num_epochs, self.learning_rate)
            elif model_type == "LSTM-regressor":
                if len(data_dict["x_train"].shape) == 2:
                    data_dict["x_train"].reshape(data_dict["x_train"].shape[0], self.window_size, -1)
                print("x_train shape", data_dict["x_train"].shape)
                in_size = data_dict["x_train"].shape[2]
                print(f"in_size: {in_size}, hl_size: {self.hidden_layer_sizes[0]}")
                regressor = LSTMRegression(in_size=in_size, hl_size=self.hidden_layer_sizes[0])
                regressor, epoch = regressor.fit(data_dict["x_train"], data_dict["y_train"], 
                                                 data_dict["x_val"], data_dict["y_val"], 
                                                 self.batch_size, self.num_epochs, self.learning_rate)
                # data_dict.update({"y_predict": regressor.predict(best_model, data_dict["x_val"], data_dict["y_val"], self.batch_size)})
            # y_predict_train = regressor.predict(self.best_model, data_dict["x_train"], data_dict["y_train"], self.batch_size)
            # wandb.log({"best_epoch": epoch, 
            #            "best_r_squared_train": r2_score(data_dict["y_train"], y_predict_train),
            #            "best_mse_train": mean_squared_error(data_dict["y_train"], y_predict_train),
            #            "best_mape_train": mape_score(data_dict["y_train"], y_predict_train),
            #            "best_rmse_train": rmse_error(data_dict["y_train"], y_predict_train),
            #            "best_symmetric_mean_absolute_percentage_error_train": smape_error(data_dict["y_train"], y_predict_train)
            #           })

        # log trained model artifact – include input features description
            metadata = {"input_shape":data_dict["x_train"].shape, 
                        "target_shape":data_dict["y_train"].shape,
                        "target_shift":self.target_shift,
                        "spatial_resolution": self.spatial_resolution,
                        "temporal_resolution": self.temporal_resolution,
                        "input_artifacts": self.input_artifacts,
                        "met_variables": self.met_variables,
                        "income_variables": self.income_variables,
                        "laqn_variables": self.laqn_variables,
                        "ablation_features": self.ablation_features}
            if model_type == "MLP-regressor":
                metadata.update({"layer_sizes": hidden_layer_sizes})
            elif model_type == "LSTM-regressor":
                metadata.update({"layer_sizes": (in_size, self.hidden_layer_sizes[0])})

            model = wandb.Artifact(
                            f"{model_type}", type="model",
                            description=f"{model_type} model.",
                            metadata=metadata)
            if model_type == "linear-regressor":
                with model.new_file("model.sav", mode="wb") as file:
                    joblib.dump(regressor, file)
            elif model_type == "MLP-regressor":
                with model.new_file("model.tar", mode="wb") as file:
                    torch.save({"state_dict": regressor.state_dict(),
                    "epoch": epoch}, file)
            elif model_type == "LSTM-regressor":
                with model.new_file("model.tar", mode="wb") as file:
                    torch.save({"state_dict": regressor.state_dict(), "epoch": epoch}, file)
            run.log_artifact(model)
    
    
    def test_and_log(self):
        model_type = self.architecture.replace("_", "-")
        with wandb.init(project="AQmortality", job_type="test-regional-model", config=self.config) as run:
            data_dict = {}
            if self.val_size:
                subsets = ["train", "val", "test"]
            else:
                subsets = ["train", "test"]
            for subset in subsets:
                data_artifact = run.use_artifact(f"xy_{subset}:latest")
                data_folder = data_artifact.download()
                file = f"{subset}.npz"
                data = np.load(path.join(data_folder, file), allow_pickle=True)
                data_dict.update({"x_"+subset: data["x"], "y_"+subset: data["y"], subset+"_dates": data["z"]})
            # use trained model artifact
            model_artifact = run.use_artifact(f"{model_type}:latest")
            model_folder = model_artifact.download()
            if model_type == "linear-regressor":
                regressor = joblib.load(path.join(model_folder, "model.sav"))
                for subset in subsets:
                    x_subset = data_dict[f"x_{subset}"].reshape(data_dict[f"x_{subset}"].shape[0], -1)
                    data_dict.update({f"y_{subset}_predict": regressor.predict(x_subset)})
    
                wandb.log({"r_squared_test": r2_score(data_dict["y_test"], data_dict["y_test_predict"]), 
                           "mse_test": mean_squared_error(data_dict["y_test"], data_dict["y_test_predict"]), 
                           "mape_test": mape_score(data_dict["y_test"], data_dict["y_test_predict"]),
                           "rmse_test": rmse_error(data_dict["y_test"], data_dict["y_test_predict"]),
                           "smape_test": smape_error(data_dict["y_test"], data_dict["y_test_predict"])
                          })
            elif model_type == "MLP-regressor":
                if len(data_dict["x_test"].shape) == 2:
                    hidden_layer_sizes = [data_dict["x_test"].shape[1]] + self.hidden_layer_sizes
                elif len(data_dict["x_test"].shape) == 3:
                    hidden_layer_sizes = [data_dict["x_test"].shape[1] * data_dict["x_test"].shape[2]] + self.hidden_layer_sizes
                else:
                    raise ValueError(f"Error in calculating network layer sizes. x_test has shape {data_dict['x_test'].shape}")
        
                regressor = MLPRegression(hidden_layer_sizes)
                checkpoint = torch.load(path.join(model_folder, "model.tar"))
                regressor.evaluate(checkpoint, data_dict["x_test"], data_dict["y_test"], self.batch_size)
                for subset in subsets:
                    data_dict.update({f"y_{subset}_predict": regressor.predict(checkpoint, data_dict[f"x_{subset}"], data_dict[f"y_{subset}"], self.batch_size)})

            elif model_type == "LSTM-regressor":
                if len(data_dict["x_test"].shape) == 2:
                    data_dict["x_test"].reshape(data_dict["x_test"].shape[0], self.window_size, -1)
                print("x_test shape", data_dict["x_test"].shape)
                in_size = data_dict["x_test"].shape[2]
                print(f"in_size: {in_size}, hl_size: {self.hidden_layer_sizes[0]}")
                hidden_layer_size = self.hidden_layer_sizes[0]  # Assuming LSTM has one hidden layer size
                regressor = LSTMRegression(in_size, hidden_layer_size, 1)
                checkpoint = torch.load(path.join(model_folder, "model.tar"))
                regressor.evaluate(checkpoint, data_dict["x_test"], data_dict["y_test"], self.batch_size)
                for subset in subsets:
                    data_dict.update({f"y_{subset}_predict": regressor.predict(checkpoint, data_dict[f"x_{subset}"], data_dict[f"y_{subset}"], self.batch_size)})


            # Save data_dict with wandb artifacts for future use.
            all_data = wandb.Artifact(
                            f"xy_all", type="dataset",
                            description=f"Input features (normalised), targets and {model_type} model predictions.",
                            metadata={"regressor_model": self.architecture, 
                                      "data_keys":list(data_dict.keys()),
                                      "target_shift":self.target_shift,
                                      "spatial_resolution": self.spatial_resolution,
                                      "temporal_resolution": self.temporal_resolution,
                                      "input_artifacts": self.input_artifacts,
                                      "met_variables": self.met_variables,
                                      "income_variables": self.income_variables})
            
            for key in data_dict.keys():
                with all_data.new_file(key+".npy", mode="wb") as file:
                    np.save(file, data_dict[key])
            run.log_artifact(all_data)
        
        return data_dict
    
    def read_data(self, artifact, version):
        with wandb.init(project="AQmortality", job_type="read-data") as run:
            data_artifact = run.use_artifact(f"{artifact}:{version}")
            data_folder = data_artifact.download()
            data_dict = {}
            for file in listdir(data_folder):
                array = np.load(path.join(data_folder, file))
                key = file.replace(".npy", "")
                data_dict.update({key: array})
        return data_dict
    

    def create_baseline(self):
        with wandb.init(project="AQmortality", job_type="baseline-model", config=self.config) as run:
            data_dict = {}
            subsets = ["test"]  # Only need test data for baseline model

            for subset in subsets:
                data_artifact = run.use_artifact(f"xy_{subset}:latest")
                data_folder = data_artifact.download()
                file = f"{subset}.npz"
                data = np.load(path.join(data_folder, file), allow_pickle=True)
                data_dict.update({f"x_{subset}": data["x"], f"y_{subset}": data["y"], subset+"_dates": data["z"]})

            # Shift mortality data for test set by the desired time lag
                if self.target_shift > 0:
                    data_dict["y_test_baseline"] = data_dict["y_test"][:-self.target_shift]
                elif self.target_shift == 0 :
                    data_dict["y_test_baseline"] = data_dict["y_test"]
            data_dict["y_test"] = data_dict["y_test"][self.target_shift:]

            # Evaluate the performance of the baseline model
            r_squared = r2_score(data_dict["y_test"], data_dict["y_test_baseline"])
            mse = mean_squared_error(data_dict["y_test"], data_dict["y_test_baseline"])
            mape = mape_score(data_dict["y_test"], data_dict["y_test_baseline"])
            rmse = rmse_error(data_dict["y_test"], data_dict["y_test_baseline"])
            smape = smape_error(data_dict["y_test"], data_dict["y_test_baseline"])

            # Log evaluation metrics
            wandb.log({
                "baseline_r_squared": r_squared,
                "baseline_mse": mse,
                "baseline_mape": mape,
                "baseline_rmse": rmse,
                "baseline_smape": smape
            })

            # Save baseline predictions with wandb artifacts for future use
            baseline_data = wandb.Artifact(
                "baseline_predictions", type="dataset",
                description="Baseline predictions using time lagged mortality data. The target mortality values in the test set are predicted as the same values from the time lagged day before.",
                metadata={
                    "target_shift": self.target_shift
                    }
            )

            for key, value in data_dict.items():
                if "y_test_baseline" in key:
                    with baseline_data.new_file(key+".npy", mode="wb") as file:
                        np.save(file, value)

            run.log_artifact(baseline_data)

# Function to create windowed input features, shifted from the target date
def create_windowed_features(df, target_shift, window_size):
    index = pd.DatetimeIndex(df.index[window_size + target_shift - 1:])
    windowed_data = np.array([np.array(df.iloc[i:i+window_size].values) for i in range(len(df) - window_size - target_shift + 1)])
    return index, windowed_data

# Metrics functions

def mape_score(targets, predictions):
    zero_indices = np.where(targets == 0)[0]  # Get the array of indices directly
    targets_drop_zero = np.delete(targets, zero_indices)
    prediction_drop_zero = np.delete(predictions, zero_indices)
    mape = np.sum(np.abs(targets_drop_zero - prediction_drop_zero) / targets_drop_zero) * 100 / len(targets_drop_zero)
    return mape

def rmse_error(targets, predictions):
    """
    Calculate the Root Mean Squared Error (RMSE) between observed and predicted values.

    Parameters:
    - targets: List, array, or other iterable of true values.
    - predictions: List, array, or other iterable of predicted values.

    Returns:
    - RMSE value.
    """
    return np.sqrt(np.mean(np.square(np.array(targets) - np.array(predictions))))

def smape_error(targets, predictions):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between observed and predicted values.

    Parameters:
    - targets: List, array, or other iterable of true values.
    - predictions: List, array, or other iterable of predicted values.

    Returns:
    - SMAPE value.
    """
    targets, predictions = np.array(targets), np.array(predictions)
    return 100 * np.mean(2 * np.abs(targets - predictions) / (np.abs(targets) + np.abs(predictions) + np.finfo(float).eps))

    

# MLP classes.

class MLPRegression():
    def __init__(self, hidden_layer_sizes, output_layer_size=1):
        self.hl_sizes = hidden_layer_sizes
        self.out_sizes = output_layer_size
        # Initialise the MLPArchitecture class
        self.model = MLPArchitecture(self.hl_sizes, self.out_sizes)
        self.metrics_functions = {"r2": r2_score, "mse": mean_squared_error, "mape": mape_score, "smape": smape_error, "rmse": rmse_error}
        
    def fit(self, x_train, y_train, x_val, y_val, batch_size, num_epochs, learning_rate, noise_standard_deviation=False):
        # Training code that loops through epochs
        model = self.model
        
        training_dataset = MLPDataset(x_train, y_train, noise_std=noise_standard_deviation)
        validation_dataset = MLPDataset(x_val, y_val)

        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimiser = Adam(model.parameters(), lr=learning_rate)
        
        wandb.watch(model, criterion, log="all", log_freq=10)
        
        training_loss_history = []
        validation_loss_history = []
        metrics_scores = {}

        for epoch in tqdm(range(num_epochs)):
            # Training set
            model.train()
            loss_sum = 0  # for storing
            y_pred_epoch = np.zeros_like(y_train)

            for batch_num, data in enumerate(training_dataloader):
                inputs_training = data["inputs"]
                targets_training = data["targets"]
                optimiser.zero_grad()

                # Run the forward pass
                inputs_training = inputs_training.view(inputs_training.size(0), -1)
                y_predict = model(inputs_training)
                # print("y_predict", y_predict.shape)
                y_pred_epoch[batch_num*batch_size : (batch_num+1)*batch_size] = y_predict.detach().numpy()
                # Compute the loss and gradients
                single_loss = criterion(torch.squeeze(y_predict), torch.squeeze(targets_training))
                single_loss.backward()
                # Update the parameters
                optimiser.step()

                # Calculate loss for storing
                loss_sum += single_loss.item()*data["targets"].shape[0]  # Account for different batch size with final batch

            training_loss_history.append(loss_sum / len(training_dataset))  # Save the training loss after every epoch
            
            # Do the same for the validation set
            model.eval()
            validation_loss_sum = 0
            y_pred_val_epoch = np.zeros_like(y_val)
            with torch.no_grad():
                for batch_num, data in enumerate(validation_dataloader):
                    inputs_val = data["inputs"]
                    targets_val = data["targets"]
                    inputs_val = inputs_val.view(inputs_val.size(0), -1)
                    y_predict_validation = model(inputs_val)
                    y_pred_val_epoch[batch_num*batch_size : (batch_num+1)*batch_size] = y_predict_validation.detach().numpy()
                    single_loss = criterion(torch.squeeze(y_predict_validation), torch.squeeze(targets_val))
                    validation_loss_sum += single_loss.item()*data["targets"].shape[0]
                    
            for metric in ["r2", "mse", "mape", "smape", "rmse"]:
                metrics_scores.update({f"{metric}_train": self.metrics_functions[metric](y_train, y_pred_epoch)})
                metrics_scores.update({f"{metric}_val": self.metrics_functions[metric](y_val, y_pred_val_epoch)})
                
            # Store the model with smallest validation loss. Check if the validation loss is the lowest BEFORE
            # saving it to loss history (otherwise it will not be lower than itself)
            if (not validation_loss_history) or validation_loss_sum / len(validation_dataset) < min(validation_loss_history):
                self.best_model = deepcopy(model)
                best_epoch = epoch
                best_metrics = metrics_scores.copy()
           
            validation_loss_history.append(validation_loss_sum / len(validation_dataset))  # Save the val loss every epoch.


            wandb.log({"training_loss": loss_sum / len(training_dataset),
                      "validation_loss": validation_loss_sum / len(validation_dataset), 
                      "r_squared_train": metrics_scores["r2_train"],
                      "r_squared_val": metrics_scores["r2_val"],
                      "mse_train": metrics_scores["mse_train"],
                      "mse_val": metrics_scores["mse_val"],
                      "mape_train": metrics_scores["mape_train"],
                       "mape_val": metrics_scores["mape_val"],
                       "smape_train": metrics_scores["smape_train"], 
                       "smape_val": metrics_scores["smape_val"], 
                       "rmse_train": metrics_scores["rmse_train"], 
                       "rmse_val": metrics_scores["rmse_val"]
                      },
                      step=epoch)
        
        wandb.log({"best_epoch": best_epoch, 
                   "best_r_squared_train": best_metrics["r2_train"],
                   "best_r_squared_val": best_metrics["r2_val"],
                   "best_mse_train": best_metrics["mse_train"],
                   "best_mse_val": best_metrics["mse_val"],
                   "best_mape_train": best_metrics["mape_train"],
                   "best_mape_val": best_metrics["mape_val"], 
                   "best_smape_train": best_metrics["smape_train"], 
                   "best_smape_val": best_metrics["smape_val"], 
                   "best_rmse_train": best_metrics["rmse_train"], 
                   "best_rmse_val": best_metrics["rmse_val"]
                  })
        
        return  self.best_model, best_epoch
    
    def evaluate(self, checkpoint, x_test, y_test, batch_size, noise_standard_deviation=False):
        model = self.model
        model.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        
        test_dataset = MLPDataset(x_test, y_test, noise_std=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        metrics_scores = {}
        
        model.eval()
        y_predict = np.zeros_like(y_test)
        with torch.no_grad():
            for batch_num, data in enumerate(test_dataloader):
                inputs = data["inputs"]
                targets = data["targets"]
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                y_predict[batch_num*batch_size : (batch_num+1)*batch_size] = outputs.detach().numpy()
                
        for metric in ["r2", "mse", "mape", "smape", "rmse"]:
            metrics_scores.update({f"{metric}_test": self.metrics_functions[metric](y_test, y_predict)})
    
        wandb.log({"r_squared_test": metrics_scores["r2_test"],
                   "mse_test": metrics_scores["mse_test"],
                   "mape_test": metrics_scores["mape_test"], 
                   "smape_test": metrics_scores["smape_test"], 
                   "rmse_test": metrics_scores["rmse_test"]
                   })
        
    
    def predict(self, checkpoint, x, y, batch_size):
        model = self.model
        model.load_state_dict(checkpoint["state_dict"])
        dataset = MLPDataset(x, y, noise_std=False)
        dataloader = DataLoader(dataset, batch_size=batch_size)
               
        model.eval()
        y_predict = np.zeros_like(y)
        with torch.no_grad():
            for batch_num, data in enumerate(dataloader):
                inputs = data["inputs"]
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                y_predict[batch_num*batch_size : (batch_num+1)*batch_size] = outputs.detach().numpy()
        return y_predict
        
    
class MLPArchitecture(nn.Module):
    def __init__(self, hl_sizes, out_size):
        super(MLPArchitecture, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hl_sizes)-1):
            self.hidden_layers.append(nn.Linear(hl_sizes[i], hl_sizes[i + 1], bias=False))
        self.output_layer = nn.Linear(hl_sizes[-1], out_size)
        
    def forward(self, x):
        
        # Check if the input size matches the expected size
        expected_size = x.shape[0] * x.shape[1]

        # Ensure the reshaping operation is correct
        if expected_size == x.numel():
            x = x.view(x.shape[0], -1)
        else:
            raise ValueError(f"Input size mismatch. Cannot reshape {x.shape} to ({x.shape[0]}, -1)")
        
        # Feedforward
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        return self.output_layer(x)
       
class MLPDataset(Dataset):
    def __init__(self, x, y, noise_std=False):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()[:, None]
        self.noise_std = noise_std  # Standard deviation of Gaussian noise

    def __len__(self):
        return self.x.size()[0]

    def nfeatures(self):
        return self.x.size()[-1]

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.noise_std:
            noise = torch.randn_like(x1)*self.noise_std
            x1 = x1 + noise
        return {"inputs": x1, "targets": y1}

# LSTM classes

class LSTMRegression():
    def __init__(self, in_size, hl_size, out_size=1):
        self.model = LSTMArchitecture(in_size, hl_size, out_size)
        self.metrics_functions = {"r2": r2_score, "mse": mean_squared_error, "mape": mape_score, "smape": smape_error, "rmse": rmse_error}

    def fit(self, x_train, y_train, x_val, y_val, batch_size, num_epochs, learning_rate):
        model = self.model

        training_dataset = MLPDataset(x_train, y_train)
        validation_dataset = MLPDataset(x_val, y_val)

        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimiser = Adam(model.parameters(), lr=learning_rate)

        wandb.watch(model, criterion, log="all", log_freq=10)

        training_loss_history = []
        validation_loss_history = []
        metrics_scores = {}

        for epoch in tqdm(range(num_epochs)):
            # Training set
            model.train()
            loss_sum = 0
            y_pred_epoch = np.zeros_like(y_train)

            for batch_num, data in enumerate(training_dataloader):
                inputs_training = data["inputs"]
                targets_training = data["targets"]
                optimiser.zero_grad()

                # Run the forward pass
                y_predict = model(inputs_training)
                y_pred_epoch[batch_num*batch_size : (batch_num+1)*batch_size] = np.expand_dims(y_predict.detach().numpy(), axis=1)
                
                # Compute the loss and gradients
                single_loss = criterion(torch.squeeze(y_predict), torch.squeeze(targets_training))
                single_loss.backward()
                
                # Update the parameters
                optimiser.step()

                # Calculate loss for storing
                loss_sum += single_loss.item() * data["targets"].shape[0]

            training_loss_history.append(loss_sum / len(training_dataset))  # Save the training loss after every epoch
            
            # Validation set
            model.eval()
            validation_loss_sum = 0
            y_pred_val_epoch = np.zeros_like(y_val)
            with torch.no_grad():
                for batch_num, data in enumerate(validation_dataloader):
                    inputs_val = data["inputs"]
                    targets_val = data["targets"]
                    y_predict_validation = model(inputs_val)
                    y_pred_val_epoch[batch_num*batch_size : (batch_num+1)*batch_size] = np.expand_dims(y_predict_validation.detach().numpy(), axis=1)
                    single_loss = criterion(torch.squeeze(y_predict_validation), torch.squeeze(targets_val))
                    validation_loss_sum += single_loss.item() * data["targets"].shape[0]
                    
            for metric in ["r2", "mse", "mape", "smape", "rmse"]:
                metrics_scores.update({f"{metric}_train": self.metrics_functions[metric](y_train, y_pred_epoch)})
                metrics_scores.update({f"{metric}_val": self.metrics_functions[metric](y_val, y_pred_val_epoch)})
                
            # Store the model with smallest validation loss
            if (not validation_loss_history) or validation_loss_sum / len(validation_dataset) < min(validation_loss_history):
                self.best_model = deepcopy(model)
                best_epoch = epoch
                best_metrics = metrics_scores.copy()
           
            validation_loss_history.append(validation_loss_sum / len(validation_dataset))

            wandb.log({"training_loss": loss_sum / len(training_dataset),
                      "validation_loss": validation_loss_sum / len(validation_dataset), 
                      "r_squared_train": metrics_scores["r2_train"],
                      "r_squared_val": metrics_scores["r2_val"],
                      "mse_train": metrics_scores["mse_train"],
                      "mse_val": metrics_scores["mse_val"],
                      "mape_train": metrics_scores["mape_train"],
                       "mape_val": metrics_scores["mape_val"],
                       "smape_train": metrics_scores["smape_train"], 
                       "smape_val": metrics_scores["smape_val"], 
                       "rmse_train": metrics_scores["rmse_train"], 
                       "rmse_val": metrics_scores["rmse_val"]
                      },
                      step=epoch)
        
        wandb.log({"best_epoch": best_epoch, 
                   "best_r_squared_train": best_metrics["r2_train"],
                   "best_r_squared_val": best_metrics["r2_val"],
                   "best_mse_train": best_metrics["mse_train"],
                   "best_mse_val": best_metrics["mse_val"],
                   "best_mape_train": best_metrics["mape_train"],
                   "best_mape_val": best_metrics["mape_val"], 
                   "best_smape_train": best_metrics["smape_train"], 
                   "best_smape_val": best_metrics["smape_val"], 
                   "best_rmse_train": best_metrics["rmse_train"], 
                   "best_rmse_val": best_metrics["rmse_val"]
                  })
        
        return  self.best_model, best_epoch
    
    def evaluate(self, checkpoint, x_test, y_test, batch_size):
        model = self.model
        model.load_state_dict(checkpoint["state_dict"])
        
        test_dataset = MLPDataset(x_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        metrics_scores = {}
        
        model.eval()
        y_predict = np.zeros_like(y_test)
        with torch.no_grad():
            for batch_num, data in enumerate(test_dataloader):
                inputs = data["inputs"]
                targets = data["targets"]
                outputs = model(inputs)
                y_predict[batch_num*batch_size : (batch_num+1)*batch_size] = np.expand_dims(outputs.detach().numpy(), axis=1)
                
        for metric in ["r2", "mse", "mape", "smape", "rmse"]:
            metrics_scores.update({f"{metric}_test": self.metrics_functions[metric](y_test, y_predict)})
    
        wandb.log({"r_squared_test": metrics_scores["r2_test"],
                   "mse_test": metrics_scores["mse_test"],
                   "mape_test": metrics_scores["mape_test"], 
                   "smape_test": metrics_scores["smape_test"], 
                   "rmse_test": metrics_scores["rmse_test"]
                   })
        
    def predict(self, checkpoint, x, y, batch_size):
        model = self.model
        model.load_state_dict(checkpoint["state_dict"])
        dataset = MLPDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size)
               
        model.eval()
        y_predict = np.zeros_like(y)
        with torch.no_grad():
            for batch_num, data in enumerate(dataloader):
                inputs = data["inputs"]
                outputs = model(inputs)
                y_predict[batch_num*batch_size : (batch_num+1)*batch_size] = np.expand_dims(outputs.detach().numpy(), axis=1)
        return y_predict

class LSTMArchitecture(nn.Module):
    def __init__(self, in_size, hl_size, out_size):
        super().__init__()
        self.hidden_layer_size = hl_size
        self.lstm = nn.LSTM(in_size, hl_size, batch_first=True)
        self.linear = nn.Linear(hl_size, out_size)

    def forward(self, input_seq):
        lstm_out, hidden_state_cell_state = self.lstm(input_seq)
        prediction = self.linear(lstm_out[:, -1, :]).reshape(-1)
        return prediction