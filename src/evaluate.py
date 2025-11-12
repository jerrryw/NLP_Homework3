import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.utils import compute_metrics

# evaluation and csv logging
def evaluate_and_log(model, test_loader, config_dict, device, results_csv_path="results/metrics.csv"):
    accuracy_score, f1_score_value = compute_metrics(model, test_loader, device)
    print(f"Test Accuracy: {accuracy_score:.4f} | F1: {f1_score_value:.4f}")

    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

    # write results into csv file
    with open(results_csv_path, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow\
        ([
            config_dict["architecture"],
            config_dict["activation"],
            config_dict["optimizer"],
            config_dict["sequence_length"],
            config_dict["gradient_clip"],
            f"{accuracy_score:.4f}",
            f"{f1_score_value:.4f}",
            ""
        ])

# accuracy/f1 vs sequence length
def plot_accuracy_f1_vs_sequence_length(results_csv_path="results/metrics.csv"):
    results_df = pd.read_csv(results_csv_path)
    if results_df.empty or "SequenceLength" not in results_df.columns:
        print("No results available for plotting.")
        return

    # convert to string
    results_df["GradientClip"] = results_df["GradientClip"].astype(str)

    # iterate through architecture types
    for architecture_name in results_df["Architecture"].unique():
        arch_df = results_df[results_df["Architecture"] == architecture_name]

        plt.figure(figsize=(8, 5))

        # plot each activation
        for activation_name in arch_df["Activation"].unique():
            sub_df = arch_df[arch_df["Activation"] == activation_name]
            means_df = sub_df.groupby("SequenceLength")[["Accuracy", "F1"]].mean()

            plt.plot(means_df.index, means_df["Accuracy"], marker="o", linestyle="--", label=f"{activation_name} (Accuracy)")
            plt.plot(means_df.index, means_df["F1"], marker="x", linestyle="-", label=f"{activation_name} (F1)")

        plt.title(f"{architecture_name} - Accuracy / F1 vs Sequence Length")
        plt.xlabel("Sequence Length")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"results/plots/{architecture_name}_accuracy_f1_vs_seq.png", dpi=200)
        plt.close()
        print(f"results/plots/{architecture_name}_accuracy_f1_vs_seq.png")

# plot loss curves for best and worst models
def plot_loss_curves(training_history_map):
    if not training_history_map:
        print("No training history for plotting")
        return

    # pick best/worst by max training accuracy
    max_accuracy_by_config = {config: max(history["acc"]) for config, history in training_history_map.items()}
    best_config            = max(max_accuracy_by_config, key=max_accuracy_by_config.get)
    worst_config           = min(max_accuracy_by_config, key=max_accuracy_by_config.get)

    # plot
    for label, config_name in [("best", best_config), ("worst", worst_config)]:
        history = training_history_map[config_name]

        plt.figure(figsize=(8, 5))
        plt.plot(history["epochs"], history["loss"], marker="o", label="Train Loss")
        plt.title(f"Training Loss vs Epochs ({label.title()} model)\n{config_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"results/plots/loss_curve_{label}.png", dpi=200)
        plt.close()
        print( f"results/plots/loss_curve_{label}.png")

# optimizer performance vs sequence length
def plot_optimizer_performance(results_csv_path="results/metrics.csv"):
    results_df = pd.read_csv(results_csv_path)
    if results_df.empty or "Optimizer" not in results_df.columns:
        print("No results for plotting.")
        return

    # group by optimizer and sequence length, compute mean metrics
    grouped_df = results_df.groupby(["Optimizer", "SequenceLength"])[["Accuracy", "F1"]].mean().reset_index().sort_values(["Optimizer", "SequenceLength"])

    plt.figure(figsize=(8, 5))
    for optimizer_name in grouped_df["Optimizer"].unique():
        optimizer_df = grouped_df[grouped_df["Optimizer"] == optimizer_name]
        plt.plot(optimizer_df["SequenceLength"], optimizer_df["Accuracy"], marker="o", linestyle="--", label=f"{optimizer_name} (Accuracy)")
        plt.plot(optimizer_df["SequenceLength"], optimizer_df["F1"], marker="x", linestyle="-", label=f"{optimizer_name} (F1)")

    plt.title("Optimizer Performance vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/optimizer_performance_vs_seq.png", dpi=200)
    plt.close()
    print("results/plots/optimizer_performance_vs_seq.png")

# function call for all
def generate_all_plots(results_csv_path="results/metrics.csv", training_history_map=None):
    plot_accuracy_f1_vs_sequence_length(results_csv_path)
    plot_optimizer_performance(results_csv_path)
    if training_history_map is not None:
        plot_loss_curves(training_history_map)

# for debug purposes
# if __name__ == "__main__":
#     generate_all_plots()
