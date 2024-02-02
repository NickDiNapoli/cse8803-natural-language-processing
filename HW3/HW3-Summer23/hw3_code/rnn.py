import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab, num_classes):
        '''
        Initialize RNN with the embedding layer, bidirectional RNN layer and a linear layer with a dropout.
    
        Args:
        vocab: Vocabulary.
        num_classes: Number of classes (labels).
        
        '''
        super(RNN, self).__init__()
        self.embed_len = 50  # embedding_dim default value for embedding layer
        self.hidden_dim = 75 # hidden_dim default value for rnn layer
        self.n_layers = 1    # num_layers default value for rnn

        vocab_size = len(vocab)
        #print(vocab)
        self.embedding = nn.Embedding(vocab_size, self.embed_len)
        self.rnn = nn.RNN(self.embed_len, self.hidden_dim, num_layers=self.n_layers, bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(.5)

    def forward(self, inputs, inputs_len):
        '''
        Implement the forward function to feed the input through the model and get the output.

        Args:
        inputs: Input data.
        inputs_len: Length of inputs in each batch.

        You can implement the forward( ) of this model by following these steps:

        1. Pass the input sequences through the embedding layer to obtain the embeddings.
        2. Pack the input sequence embeddings, and then pass it through the RNN layer to get the output from the RNN layer, which should be padded.
        3. Concatenate the first hidden state in the reverse direction and the last hidden state in the forward direction of the bidirectional RNN and pass it to the linear layer. 
                Take a look at the architecture diagram of our model in HW3.ipynb to visually see how this is done.

        HINTS : 
            1. For packing and padding sequences, consider using : torch.nn.utils.rnn.pack_padded_sequence and torch.nn.utils.rnn.pad_packed_sequence. Set 'batch_first' = True and enforce_sorted = False (for packing)
            2. Refer to https://pytorch.org/docs/stable/generated/torch.nn.RNN.html to see what the output of the RNN looks like. This may be helpful for Step 3.
            3. Using dropout layers can also help in improving accuracy.

        Returns:
            output: Logits of each label.
        '''

        embedding = self.embedding(inputs)
        embedding = pack_padded_sequence(embedding, inputs_len, batch_first=True, enforce_sorted=False)

        output, h_n = self.rnn(embedding)
        output, h_n = pad_packed_sequence(output, batch_first=True)
        #print(output.shape)
        output = torch.cat((output[:, -1, :self.hidden_dim], output[:, 0, self.hidden_dim:]), dim=1)

        output = self.dropout(output)
        output = self.linear(output)
        return output
