import os
import argparse
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch
import scipy.stats as stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from model_classes import MLPRegression, LSTMRegression, mape_score, smape_error


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, default="AQmortality")
    parser.add_argument("--entity", type=str, default="michellewl")

    parser.add_argument("--model", type=str, choices=["baseline", "linear", "MLP", "LSTM"], required=True)
    parser.add_argument("--model_artifact", type=str, required=False)

    parser.add_argument("--output_dir", type=str, default="results/corrections")

    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--target_shift", type=int, default=1)
    parser.add_argument("--window_size", type=int, required=True)

    parser.add_argument("--hidden_layer_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--laqn_version", type=str, default="v7")
    parser.add_argument("--met_version", type=str, default="v3")
    parser.add_argument("--income_version", type=str, default="v3")
    parser.add_argument("--mortality_version", type=str, default="v1")

    return parser.parse_args()


def create_windowed_features(df, target_shift, window_size):
    index = pd.DatetimeIndex(df.index[window_size + target_shift - 1:])
    windowed_data = np.array([
        np.array(df.iloc[i:i + window_size].values)
        for i in range(len(df) - window_size - target_shift + 1)
    ])
    return index, windowed_data


def rebuild_test_data(run, args):
    laqn_variables = ["NO2", "PM10"]
    met_variables = ["temperature", "humidity", "dew_point", "wind_speed", "pressure"]
    income_variables = [
        "gross_disposable_income",
        "property_income_received",
        "social_benefits_received",
        "current_taxes_on_wealth",
    ]

    df_list = []

    # LAQN regional data
    laqn_artifact = run.use_artifact(f"laqn-regional:{args.laqn_version}")
    laqn_folder = laqn_artifact.download()

    for variable in laqn_variables:
        for file in os.listdir(laqn_folder):
            if variable in file and file.endswith(".npz"):
                data = np.load(os.path.join(laqn_folder, file), allow_pickle=True)
                column = file.replace(".npz", "")
                df_list.append(
                    pd.DataFrame(
                        index=pd.DatetimeIndex(data["x"]),
                        data=data["y"],
                        columns=[column],
                    )
                )

    # Meteorology data
    met_artifact = run.use_artifact(f"met-resample:{args.met_version}")
    met_folder = met_artifact.download()

    for variable in met_variables:
        filepath = os.path.join(met_folder, f"{variable}.npz")
        data = np.load(filepath, allow_pickle=True)

        variable_df = pd.concat(
            [
                pd.Series(data["mean"], index=data["x"], name=f"{variable}_mean"),
                pd.Series(data["min"], index=data["x"], name=f"{variable}_min"),
                pd.Series(data["max"], index=data["x"], name=f"{variable}_max"),
            ],
            axis=1,
        )
        variable_df.index = pd.DatetimeIndex(variable_df.index)
        df_list.append(variable_df)

    # Income data
    income_artifact = run.use_artifact(f"income-regional:{args.income_version}")
    income_folder = income_artifact.download()

    for variable in income_variables:
        for file in os.listdir(income_folder):
            if variable in file and file.endswith(".npz"):
                data = np.load(os.path.join(income_folder, file), allow_pickle=True)
                column = file.replace(".npz", "")
                df_list.append(
                    pd.DataFrame(
                        index=pd.DatetimeIndex(data["x"]),
                        data=data["y"],
                        columns=[column],
                    )
                )

    df = pd.concat(df_list, axis=1)
    input_columns = df.columns
    df = df.dropna(axis=0)

    # Mortality target
    mortality_artifact = run.use_artifact(f"mortality-scaled:{args.mortality_version}")
    mortality_folder = mortality_artifact.download()

    mortality_data = np.load(
        os.path.join(mortality_folder, "deaths.npz"),
        allow_pickle=True,
    )

    mortality_df = pd.DataFrame(
        index=pd.DatetimeIndex(mortality_data["x"]),
        data=mortality_data["y"] * 100000,
        columns=["deaths"],
    )

    df = df.join(mortality_df).dropna(axis=0)

    train_end = int(len(df.index) * args.train_size)
    test_start = int(len(df.index) * (args.train_size + args.val_size))

    train_index = df.index[:train_end]
    test_index = df.index[test_start:]

    scaler = MinMaxScaler()
    scaler.fit(df.loc[train_index].drop("deaths", axis=1))

    x_test_raw = scaler.transform(df.loc[test_index].drop("deaths", axis=1))
    y_test_raw = df.loc[test_index]["deaths"].values
    test_dates_raw = df.loc[test_index].index

    x_test_df = pd.DataFrame(
        index=pd.DatetimeIndex(test_dates_raw),
        data=x_test_raw,
        columns=input_columns,
    )

    y_test_df = pd.DataFrame(
        index=pd.DatetimeIndex(test_dates_raw),
        data=y_test_raw,
        columns=["deaths"],
    )

    test_dates, x_test = create_windowed_features(
        x_test_df,
        args.target_shift,
        args.window_size,
    )

    y_test = y_test_df.loc[test_dates].values

    return x_test, y_test, test_dates


def load_model_folder(run, model_artifact):
    artifact = run.use_artifact(model_artifact)
    return artifact.download()


def predict(model_folder, x_test, y_test, args):
    if args.model == "linear":
        model_path = os.path.join(model_folder, "model.sav")
        regressor = joblib.load(model_path)
        x_flat = x_test.reshape(x_test.shape[0], -1)
        y_pred = regressor.predict(x_flat)

    elif args.model == "MLP":
        if args.hidden_layer_size is None or args.num_layers is None:
            raise ValueError("For MLP, provide --hidden_layer_size and --num_layers")

        checkpoint = torch.load(
            os.path.join(model_folder, "model.tar"),
            map_location=torch.device("cpu"),
        )

        if len(x_test.shape) == 2:
            input_size = x_test.shape[1]
        elif len(x_test.shape) == 3:
            input_size = x_test.shape[1] * x_test.shape[2]
        else:
            raise ValueError(f"Unexpected x_test shape for MLP: {x_test.shape}")

        hidden_layer_sizes = [input_size] + [args.hidden_layer_size] * args.num_layers
        regressor = MLPRegression(hidden_layer_sizes)
        y_pred = regressor.predict(checkpoint, x_test, y_test, args.batch_size)

    elif args.model == "LSTM":
        if args.hidden_layer_size is None or args.num_layers is None:
            raise ValueError("For LSTM, provide --hidden_layer_size and --num_layers")

        checkpoint = torch.load(
            os.path.join(model_folder, "model.tar"),
            map_location=torch.device("cpu"),
        )

        if len(x_test.shape) != 3:
            raise ValueError(f"Expected 3D x_test for LSTM, got shape {x_test.shape}")

        input_size = x_test.shape[2]
        regressor = LSTMRegression(
            input_size,
            args.hidden_layer_size,
            args.num_layers,
        )
        y_pred = regressor.predict(checkpoint, x_test, y_test, args.batch_size)

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return np.asarray(y_pred).flatten()


def baseline_predict(y_test, target_shift):
    y_test = y_test.flatten()

    if target_shift > 0:
        y_true = y_test[target_shift:]
        y_pred = y_test[:-target_shift]
    else:
        y_true = y_test
        y_pred = y_test

    return y_true, y_pred

def compute_metrics(y_true, y_pred):
    residuals = y_true - y_pred

    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mape_score(y_true, y_pred),
        "smape": smape_error(y_true, y_pred),
        "residual_mean": np.mean(residuals),
        "residual_std": np.std(residuals),
        "residual_min": np.min(residuals),
        "residual_max": np.max(residuals),
    }


def scatter_plot(y_true, y_pred, title):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    return fig


def residual_plot(y_true, y_pred, title):
    residuals = y_true - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, linestyle="--")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(title)

    return fig


def qq_plot(y_true, y_pred, title):
    residuals = y_true - y_pred

    fig, ax = plt.subplots()
    stats.probplot(residuals.flatten(), dist="norm", plot=ax)
    ax.set_title(title)

    return fig


def distribution_plot(y_true, y_pred, title):
    fig, ax = plt.subplots()

    ax.hist(y_true, bins=30, alpha=0.5, density=True, label="Observed")
    ax.hist(y_pred, bins=30, alpha=0.5, density=True, label="Predicted")

    ax.set_xlabel("Deaths")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    return fig


def timeseries_plot(dates, y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(dates, y_true, label="Observed")
    ax.plot(dates, y_pred, label="Predicted")

    ax.set_xlabel("Date")
    ax.set_ylabel("Deaths")
    ax.set_title(title)
    ax.legend()

    fig.autofmt_xdate()
    return fig


def save_metrics(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics_long = pd.DataFrame(metrics.items(), columns=["metric", "value"])
    metrics_long.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    metrics_wide = pd.DataFrame([metrics])
    metrics_wide.to_csv(os.path.join(output_dir, "metrics_wide.csv"), index=False)


def save_predictions(dates, y_true, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "observed": y_true,
            "predicted": y_pred,
            "residual": y_true - y_pred,
        }
    )

    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


def save_plot(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate(args):
    if args.model == "baseline":
        artifact_label = "no-model"
    else:
        if args.model_artifact is None:
            raise ValueError("You must provide --model_artifact unless --model baseline is used.")
        artifact_label = args.model_artifact.replace("/", "_").replace(":", "-")

    output_dir = os.path.join(
        args.output_dir,
        f"{args.model}_window-{args.window_size}_{artifact_label}",
    )

    with wandb.init(
        project=args.project,
        entity=args.entity,
        job_type="final-evaluation",
    ) as run:
        x_test, y_test, test_dates = rebuild_test_data(run, args)

        if args.model == "baseline":
            y_true, y_pred = baseline_predict(y_test, args.target_shift)
            test_dates = test_dates[args.target_shift:]
        else:
            model_folder = load_model_folder(run, args.model_artifact)
            y_true = y_test.flatten()
            y_pred = predict(model_folder, x_test, y_test, args).flatten()

        metrics = compute_metrics(y_true, y_pred)

        save_metrics(metrics, output_dir)
        save_predictions(test_dates, y_true, y_pred, output_dir)

        scatter_fig = scatter_plot(
            y_true,
            y_pred,
            f"{args.model}: Predicted vs Observed",
        )
        residual_fig = residual_plot(
            y_true,
            y_pred,
            f"{args.model}: Residual Plot",
        )
        qq_fig = qq_plot(
            y_true,
            y_pred,
            f"{args.model}: Q-Q Plot of Residuals",
        )
        dist_fig = distribution_plot(
            y_true,
            y_pred,
            f"{args.model}: Observed vs Predicted Distribution",
        )
        ts_fig = timeseries_plot(
            test_dates,
            y_true,
            y_pred,
            f"{args.model}: Observed vs Predicted Time Series",
        )

        save_plot(scatter_fig, "scatter.png", output_dir)
        save_plot(residual_fig, "residuals.png", output_dir)
        save_plot(qq_fig, "qq_plot.png", output_dir)
        save_plot(dist_fig, "distribution.png", output_dir)
        save_plot(ts_fig, "timeseries.png", output_dir)

        wandb.log(metrics)

        wandb.log(
            {
                "scatter_plot": wandb.Image(
                    scatter_plot(y_true, y_pred, f"{args.model}: Predicted vs Observed")
                ),
                "residual_plot": wandb.Image(
                    residual_plot(y_true, y_pred, f"{args.model}: Residual Plot")
                ),
                "qq_plot": wandb.Image(
                    qq_plot(y_true, y_pred, f"{args.model}: Q-Q Plot of Residuals")
                ),
                "distribution_plot": wandb.Image(
                    distribution_plot(
                        y_true,
                        y_pred,
                        f"{args.model}: Observed vs Predicted Distribution",
                    )
                ),
                "timeseries_plot": wandb.Image(
                    timeseries_plot(
                        test_dates,
                        y_true,
                        y_pred,
                        f"{args.model}: Observed vs Predicted Time Series",
                    )
                ),
            }
        )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)