import numpy as np
import random
from sklearn.metrics import f1_score
import torch

# evaluate a trained model on dataLoader and return accuracy and macro-F1.
def compute_metrics(model, data_loader, device):
    model.eval()
    predicted_labels = []
    true_labels      = []

    # disable gradient tracking
    with torch.no_grad():
        for input_batch, label_batch in data_loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.float().to(device)

            # forward pass
            logits = model(input_batch).squeeze()

            # convert to binary predictions
            batch_predictions = (logits >= 0.5).cpu().numpy()

            predicted_labels.extend(batch_predictions)
            true_labels.extend(label_batch.cpu().numpy())

    # compute results
    accuracy_score = sum(predict == true for predict, true in zip(predicted_labels, true_labels)) / len(true_labels)
    f1_score_value = f1_score(true_labels, predicted_labels, average="macro")

    return accuracy_score, f1_score_value

# set seeds for reproducibility
def set_seed(seed_value: int = 42):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
