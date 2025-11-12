import torch
import torch.nn as nn

class BaseRNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension = 100, hidden_dimension = 64,
                 number_of_layers = 2, dropout_probability = 0.5, rnn_type = "RNN",
                 bidirectional = False, activation_name = "sigmoid"):
        super().__init__()

        # converts token indices into dense vector representations
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=0)

        # directionality for final linear layer sizing
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # recurrent layer selection
        if rnn_type == "RNN":
            self.recurrent = nn.RNN(embedding_dimension, hidden_dimension, num_layers=number_of_layers, dropout=dropout_probability, batch_first=True)
        elif rnn_type == "LSTM":
            self.recurrent = nn.LSTM(embedding_dimension, hidden_dimension, num_layers=number_of_layers, dropout=dropout_probability, batch_first=True)
        elif rnn_type == "BiLSTM":
            self.recurrent = nn.LSTM(embedding_dimension, hidden_dimension, num_layers=number_of_layers, dropout=dropout_probability, batch_first=True, bidirectional=True)
        else:
            raise ValueError("RNN_type must be 'RNN', 'LSTM', or 'BiLSTM'")

        # activation function
        activation_map       = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}
        self.head_activation = activation_map.get(activation_name.lower(), nn.ReLU())

        # classification
        self.dropout           = nn.Dropout(dropout_probability)
        self.classifier        = nn.Linear(hidden_dimension * self.num_directions, 1)
        self.output_activation = nn.Sigmoid()

    # forward pass
    def forward(self, input_ids: torch.Tensor):
        embedded_tokens     = self.embedding(input_ids)
        sequence_outputs, _ = self.recurrent(embedded_tokens)

        # extract output from the last time step
        last_time_step      = sequence_outputs[:, -1, :]
        last_time_step      = self.dropout(last_time_step)
        last_time_step      = self.head_activation(last_time_step)

        # project down
        logits              = self.classifier(last_time_step)

        return self.output_activation(logits)
