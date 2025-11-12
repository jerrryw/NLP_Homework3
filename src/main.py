from itertools import product
import os
import pandas as pd
from time import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.evaluate import evaluate_and_log, generate_all_plots
from src.preprocess import (
    padded_train_sequences_25,
    padded_test_sequences_25,
    padded_train_sequences_50,
    padded_test_sequences_50,
    padded_train_sequences_100,
    padded_test_sequences_100,
    train_dataframe,
    test_dataframe,
    vocabulary_token_to_index,
)
from src.train import train_model
from src.utils import set_seed

# set seed for reproducibility
set_seed(42)

# CUDA acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# convert sentiment labels to tensors
train_label_tensor = torch.tensor((train_dataframe["sentiment"] == "positive").astype(int).values, dtype=torch.float32, device=device)
test_label_tensor  = torch.tensor((test_dataframe["sentiment"] == "positive").astype(int).values, dtype=torch.float32, device=device)

# map sequence lengths to corresponding padded tensors
sequence_tensor_map =\
{
    25: (padded_train_sequences_25, padded_test_sequences_25),
    50: (padded_train_sequences_50, padded_test_sequences_50),
    100: (padded_train_sequences_100, padded_test_sequences_100)
}

# full examples for training
architecture_choices    = ["RNN", "LSTM", "BiLSTM"]
activation_choices      = ["relu", "tanh", "sigmoid"]
optimizer_choices       = ["adam", "sgd", "rmsprop"]
sequence_length_choices = [25, 50, 100]
gradient_clip_choices   = [None, 1.0]

# small examples for debuging
# architecture_choices    = ["RNN", "LSTM", "BiLSTM"]
# activation_choices      = ["relu", "sigmoid"]
# optimizer_choices       = ["adam","rmsprop"]
# sequence_length_choices = [100]
# gradient_clip_choices   = [1.0]

results_csv_path = "results/metrics.csv"

# create directory if doesn't exist
os.makedirs("results", exist_ok=True)
with open(results_csv_path, "w") as csv_file:
    csv_file.write("Architecture,Activation,Optimizer,SequenceLength,GradientClip,Accuracy,F1,EpochTimeSeconds\n")

vocabulary_size = len(vocabulary_token_to_index)
num_epochs      = 5

# store training loss/accuracy history
training_history_map = {}

# combinations of experiment settings
config_grid = list(product(architecture_choices, activation_choices, optimizer_choices, sequence_length_choices, gradient_clip_choices))
total_runs  = len(config_grid)

# main loop
for run_index, (architecture, activation_name, optimizer_name, sequence_length, gradient_clip_norm) in enumerate(config_grid, start=1):
    config_string = f"{architecture}-{activation_name}-{optimizer_name}-Seq:{sequence_length}-Clip:{gradient_clip_norm}"
    print(f"\n=== [{run_index}/{total_runs}] Running {config_string} ===")

    # train/test tensors for given sequence length
    train_input_tensor, test_input_tensor = sequence_tensor_map[sequence_length]
    start_time = time()

    # train and collect history
    model, epoch_history = train_model(
        train_input_tensor=train_input_tensor,
        train_label_tensor=train_label_tensor,
        validation_input_tensor=test_input_tensor,
        validation_label_tensor=test_label_tensor,
        vocabulary_size=vocabulary_size,
        rnn_type=architecture,
        activation_name=activation_name,
        optimizer_name=optimizer_name,
        sequence_length=sequence_length,
        gradient_clip_norm=gradient_clip_norm,
        epochs=num_epochs,
        dropout_probability=0.5,
        batch_size=32,
        device=device,
        record_history=True
    )

    # total time for this combination
    elapsed_seconds                     = time() - start_time
    training_history_map[config_string] = epoch_history
    print(f"Total Elapsed Time: {elapsed_seconds:.2f}")

    # evaluate + append csv row
    config_dict =\
    {
        "architecture": architecture,
        "activation": activation_name,
        "optimizer": optimizer_name,
        "sequence_length": sequence_length,
        "gradient_clip": "Yes" if gradient_clip_norm else "No"
    }

    # dataloader for testing
    test_loader = DataLoader(TensorDataset(test_input_tensor, test_label_tensor), batch_size=32, shuffle=False)
    evaluate_and_log(model, test_loader, config_dict, device, results_csv_path)

    # udpate csv file
    df_metrics = pd.read_csv(results_csv_path)
    df_metrics.loc[df_metrics.index[-1], "EpochTimeSeconds"] = round(elapsed_seconds / num_epochs, 2)
    df_metrics.to_csv(results_csv_path, index=False)

print("\nAll experiments completed. Results saved to results/metrics.csv")

# plot
print("\nGenerating plots:")
generate_all_plots(results_csv_path, training_history_map)
print("All plots saved under results/plots")
