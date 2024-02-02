import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    def __init__(self, vocab, num_classes):
        """
        Initialize Attention with the embedding layer, LSTM layer and a linear layer.
        NOTE: 1. A context layer also needs to be defined for attention. 
              2. Adding a dropout layer would be useful
        
        Args:
            vocab: Vocabulary. (Refer to the documentation as specified in lstm.py)
            num_classes: Number of classes (labels).
        
        Doesn't return anything

        """
        super(Attention, self).__init__()

        num_embeddings = vocab.__len__()
        self.embed_len = 100
        self.embedding_layer = nn.Embedding(num_embeddings, self.embed_len)

        self.hidden_dim = 75
        self.n_layers = 1
        self.LSTM = nn.LSTM(input_size=self.embed_len, hidden_size=self.hidden_dim, num_layers=self.n_layers,
                            bidirectional=True, batch_first=True)

        self.context = nn.Linear(self.hidden_dim * 2, num_classes)

        self.linear = nn.Linear(self.hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(.5)

    def forward(self, inputs, inputs_len):
        """
        Implement the forward function to feed the input through the model and get the output.
        Args:
          inputs: Input data.
          inputs_len: Length of inputs in each batch.
        NOTE : 
            1. For padding and packing sequences, consider using : torch.nn.utils.rnn.pack_padded_sequence and torch.nn.utils.rnn.pad_packed_sequence.
            2. Using dropout layers can also help in improving accuracy.
            3. BMM operation of torch can be helpful for matrix multiplication to compute attention weights and for the context. 
                Refer to this for understanding the function usage: https://pytorch.org/docs/stable/generated/torch.bmm.html
            4. Softmax function would be useful for attention weights
        Returns:
            output: Logits of each label. The output is a tensor of shape (B, C) where B is the batch size and C is the number of classes
        """

        embedding = self.embedding_layer(inputs)
        embedding = pack_padded_sequence(embedding, inputs_len, batch_first=True, enforce_sorted=False)

        output, (h_n, c_n) = self.LSTM(embedding)
        output, h_n = pad_packed_sequence(output, batch_first=True)
        # print(output.shape)
        attention = F.softmax(self.context(output), dim=1)
        output = torch.bmm(attention.transpose(1, 2), output)

        output = torch.cat((output[:, -1, :self.hidden_dim], output[:, 0, self.hidden_dim:]), dim=1)

        output = self.dropout(output)
        output = self.linear(output)

        return output
