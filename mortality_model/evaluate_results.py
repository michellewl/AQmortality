import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

output_directory = "results/corrections"

def load_results(artifact_name="xy_all", version="latest"):
    with wandb.init(project="AQmortality", job_type="evaluate-results") as run:
        artifact = run.use_artifact(f"{artifact_name}:{version}")
        folder = artifact.download()

        data = {}
        for file in os.listdir(folder):
            key = file.replace(".npy", "")
            data[key] = np.load(os.path.join(folder, file))

    return data

def compute_metrics(y_true, y_pred):
    residuals = y_true - y_pred

    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "residual_mean": np.mean(residuals),
        "residual_std": np.std(residuals)
    }

def scatter_plot(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Observed")

    return fig

def residual_plot(y_true, y_pred):
    residuals = y_true - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, linestyle="--")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")

    return fig

def qq_plot(y_true, y_pred):
    residuals = y_true - y_pred

    fig, ax = plt.subplots()
    stats.probplot(residuals.flatten(), dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Residuals)")

    return fig

def distribution_plot(y_true, y_pred):
    fig, ax = plt.subplots()
    sns.kdeplot(y_true.flatten(), label="Observed", ax=ax)
    sns.kdeplot(y_pred.flatten(), label="Predicted", ax=ax)
    ax.legend()
    ax.set_title("Distribution Comparison")

    return fig

def save_plot(fig, filename, output_dir=output_directory):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def evaluate():
    data = load_results()

    y_true = data["y_test"].flatten()
    y_pred = data["y_test_predict"].flatten()

    metrics = compute_metrics(y_true, y_pred)

    # Save metrics locally
    save_metrics(metrics)

    # Generate plots
    scatter_fig = scatter_plot(y_true, y_pred)
    residual_fig = residual_plot(y_true, y_pred)
    qq_fig = qq_plot(y_true, y_pred)
    dist_fig = distribution_plot(y_true, y_pred)

    # Save plots locally
    save_plot(scatter_fig, "scatter.png")
    save_plot(residual_fig, "residuals.png")
    save_plot(qq_fig, "qq_plot.png")
    save_plot(dist_fig, "distribution.png")

    # Optional: still log to wandb
    with wandb.init(project="AQmortality", job_type="final-evaluation") as run:
        wandb.log(metrics)
        wandb.log({
            "scatter_plot": wandb.Image(scatter_fig),
            "residual_plot": wandb.Image(residual_fig),
            "qq_plot": wandb.Image(qq_fig),
            "distribution_plot": wandb.Image(dist_fig)
        })

def save_metrics(metrics, output_dir=output_directory):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(metrics.items(), columns=["metric", "value"])
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

if __name__ == "__main__":
    evaluate()