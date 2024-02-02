import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab, num_classes):
        """
        Initialize LSTM with the embedding layer, LSTM layer and a linear layer.
        #Hint: nn.Embedding, nn.lSTM might be useful
        #Hint: For Linear Layer, consider that it is a bi-directional LSTM
        NOTE:
              1. Use these names to create the parameters:
                For Embedding layer: embedding size should be intialized as variable "self.embed_len"
                For LSTM: Hidden dimension size: "self.hidden_dim", number of layers: "self.n_layers"
              2. For LSTM, consider only 1 layer
        Args:
            vocab: Vocabulary. (Refer to this for documentation: https://pytorch.org/text/stable/vocab.html)
            num_classes: Number of classes (labels).

        Doesn't return anything
        """
        super(LSTM, self).__init__()
        #print(type(vocab))
        num_embeddings = vocab.__len__()
        self.embed_len = 100
        self.embedding_layer = nn.Embedding(num_embeddings, self.embed_len)

        self.hidden_dim = 75
        self.n_layers = 1
        self.LSTM = nn.LSTM(input_size=self.embed_len, hidden_size=self.hidden_dim, num_layers=self.n_layers, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(self.hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(.5)

    def forward(self, inputs, inputs_len):
        """
        Implement the forward function to feed the input through the model and get the output.

        Args:
        	inputs: Input data. #Tensor of shape (B, L) where B is batch size and L is the length of the longest sentence in the data
        	inputs_len: Length of inputs in each batch.

        NOTE :
              1. For padding and packing sequences, consider using : torch.nn.utils.rnn.pack_padded_sequence and torch.nn.utils.rnn.pad_packed_sequence.
              2. Using dropout layers can also help in improving accuracy.
              3. For LSTM outputs refer to this documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html.
                (Hint: Might be useful for input to the dropout layer)
              Remember that it is a bi-directional LSTM.

        Returns:
              output: Logits of each label. The output is a tensor of shape (B, C) where B is the batch size and C is the number of classes
        """

        embedding = self.embedding_layer(inputs)
        embedding = pack_padded_sequence(embedding, inputs_len, batch_first=True, enforce_sorted=False)

        output, (h_n, c_n) = self.LSTM(embedding)
        output, h_n = pad_packed_sequence(output, batch_first=True)
        # print(output.shape)
        output = torch.cat((output[:, -1, :self.hidden_dim], output[:, 0, self.hidden_dim:]), dim=1)

        output = self.dropout(output)
        output = self.linear(output)

        return output
