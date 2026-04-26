import os
import argparse
import pandas as pd
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="michellewl")
    parser.add_argument("--project", type=str, default="AQmortality")
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/sweep_summary")
    return parser.parse_args()


def get_sweep_runs(entity, project, sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    return list(sweep.runs)


def extract_run_results(runs):
    rows = []

    metric_names = [
        "r_squared_test",
        "mse_test",
        "rmse_test",
        "mape_test",
        "smape_test",
        "best_r_squared_val",
        "best_mse_val",
        "best_rmse_val",
        "best_mape_val",
        "best_smape_val",
    ]

    config_names = [
        "model",
        "architecture",
        "num_layers",
        "hidden_layer_size",
        "window_size",
        "random_seed",
        "batch_size",
        "learning_rate",
        "num_epochs",
    ]

    for run in runs:
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "url": run.url,
        }

        for key in config_names:
            row[key] = run.config.get(key, None)

        for key in metric_names:
            row[key] = run.summary.get(key, None)

        rows.append(row)

    return pd.DataFrame(rows)


def summarise_results(df):
    group_cols = [
        "model",
        "num_layers",
        "hidden_layer_size",
        "window_size",
    ]

    available_group_cols = [c for c in group_cols if c in df.columns]

    metric_cols = [
        "r_squared_test",
        "mse_test",
        "rmse_test",
        "mape_test",
        "smape_test",
    ]

    available_metric_cols = [
        c for c in metric_cols
        if c in df.columns and df[c].notna().any()
    ]

    summary = (
        df
        .groupby(available_group_cols, dropna=False)[available_metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if x != ""])
        for col in summary.columns
    ]

    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    runs = get_sweep_runs(args.entity, args.project, args.sweep_id)
    df = extract_run_results(runs)

    raw_path = os.path.join(args.output_dir, f"{args.sweep_id}_all_runs.csv")
    df.to_csv(raw_path, index=False)

    finished = df[df["state"] == "finished"].copy()

    summary = summarise_results(finished)
    summary_path = os.path.join(args.output_dir, f"{args.sweep_id}_summary.csv")
    summary.to_csv(summary_path, index=False)

    if "mse_test" in finished.columns:
        best = finished.sort_values("mse_test", ascending=True).head(10)
        best_path = os.path.join(args.output_dir, f"{args.sweep_id}_best_runs.csv")
        best.to_csv(best_path, index=False)

    print(f"Saved raw run table to: {raw_path}")
    print(f"Saved summary table to: {summary_path}")


if __name__ == "__main__":
    main()