from time import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.models import BaseRNN
from src.utils import compute_metrics

def train_model(train_input_tensor, train_label_tensor, validation_input_tensor, validation_label_tensor,
                vocabulary_size, rnn_type = "RNN", activation_name = "relu", optimizer_name = "adam",
                sequence_length = 50, gradient_clip_norm = None, device = "cuda" if torch.cuda.is_available() else "cpu",
                epochs = 5, batch_size = 32, dropout_probability = 0.5, record_history = False):

    # init model
    model = BaseRNN(vocabulary_size=vocabulary_size, embedding_dimension=100, hidden_dimension=64, number_of_layers=2,
                    dropout_probability=dropout_probability, rnn_type=rnn_type, activation_name=activation_name,
                    bidirectional=True if rnn_type == "BiLSTM" else False).to(device)

    # binary cross entropy loss
    loss_function = nn.BCELoss()

    # choose optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    else:
        raise ValueError("Unsupported Optimizer.")

    # create dataloader for training and testing
    train_loader = DataLoader(TensorDataset(train_input_tensor, train_label_tensor), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(validation_input_tensor, validation_label_tensor), batch_size=batch_size, shuffle=False)

    history = {"epochs": [], "loss": [], "acc": [], "val_acc": [], "val_f1": []}

    # main loop
    for epoch_index in range(epochs):
        model.train()
        cumulative_train_loss   = 0.0
        num_correct_predictions = 0
        num_seen_examples       = 0
        epoch_start_time        = time()

        for input_batch, label_batch in tqdm(train_loader, desc=f"[{rnn_type}] Epoch {epoch_index + 1}/{epochs}"):
            input_batch = input_batch.to(device)
            label_batch = label_batch.float().to(device)

            # forward pass
            optimizer.zero_grad()
            probabilities = model(input_batch).squeeze()
            batch_loss    = loss_function(probabilities, label_batch)

            # backward pass
            batch_loss.backward()

            # gradient clipping
            if gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            # update weights
            optimizer.step()

            # track loss/accuracy for this batch
            cumulative_train_loss   += batch_loss.item() * input_batch.size(0)
            batch_predictions        = (probabilities >= 0.5).float()
            num_correct_predictions += (batch_predictions == label_batch).sum().item()
            num_seen_examples       += label_batch.size(0)

        train_accuracy                     = num_correct_predictions / num_seen_examples
        validation_accuracy, validation_f1 = compute_metrics(model, valid_loader, device)
        average_train_loss                 = cumulative_train_loss / num_seen_examples
        epoch_duration_seconds             = time() - epoch_start_time

        # current epoch print stmt
        print\
        (
            f"Epoch {epoch_index + 1}/{epochs} | "
            f"Loss: {average_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {validation_accuracy:.4f} | Val F1: {validation_f1:.4f} | "
            f"Time: {epoch_duration_seconds:.1f}s"
        )

        if record_history:
            history["epochs"].append(epoch_index + 1)
            history["loss"].append(average_train_loss)
            history["acc"].append(train_accuracy)
            history["val_acc"].append(validation_accuracy)
            history["val_f1"].append(validation_f1)

    if record_history:
        return model, history
    return model
