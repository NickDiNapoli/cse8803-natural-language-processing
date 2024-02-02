import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, w2vmodel, num_classes, window_sizes=(1,2,3,5)):
        '''
        Initialize CNN with the embedding layer, convolutional layer and a linear layer.

        Steps to initialize -
        1. For embedding layer, create embedding from pretrained model using w2vmodel weights vectors, and padding.
        nn.Embedding would be useful.

        2. Create convolutional layers with the given window_sizes and padding of (window_size - 1, 0).
        nn.Conv2d would be useful. Additionally, nn.ModuleList might also be useful.

        3. Linear layer with num_classes output.
        Args:
            w2vmodel: Pre-trained word2vec model.
            num_classes: Number of classes (labels).
            window_sizes: Window size for the convolution kernel.
        '''
        super(CNN, self).__init__()
        weights = w2vmodel.wv # use this to initialize the embedding layer
        EMBEDDING_SIZE = 500  # Use this to set the embedding_dim in embedding layer
        NUM_FILTERS = 10      # Number of filters in CNN
        '''
        self.module_list = nn.ModuleList()
        print(weights, type(weights))
        print(len(weights.index_to_key))
        self.embedding = nn.Embedding(len(weights.index_to_key), EMBEDDING_SIZE)
        #self.embedding = self.embedding(self.embed)
        #self.module_list.append(self.embedding)

        self.layer1 = nn.Conv2d(EMBEDDING_SIZE, NUM_FILTERS, window_sizes[0], stride=1, padding=(window_sizes[0] - 1, 0))
        self.module_list.append(self.layer1)
        self.module_list.append(nn.Tanh())
        self.module_list.append(nn.MaxPool1d(1))
        self.layer2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, window_sizes[1], stride=1, padding=(window_sizes[1] - 1, 0))
        self.module_list.append(self.layer2)
        self.module_list.append(nn.Tanh())
        self.module_list.append(nn.MaxPool1d(1))
        self.layer3 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, window_sizes[2], stride=1, padding=(window_sizes[2] - 1, 0))
        self.module_list.append(self.layer3)
        self.module_list.append(nn.Tanh())
        self.module_list.append(nn.MaxPool1d(1))
        self.layer4 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, window_sizes[3], stride=1, padding=(window_sizes[3] - 1, 0))
        self.module_list.append(self.layer4)
        self.module_list.append(nn.Tanh())
        self.module_list.append(nn.MaxPool1d(1))

        self.linear = nn.Linear(in_features=NUM_FILTERS, out_features=num_classes)
        self.module_list.append(self.linear)
        '''
        weight_vectors = torch.tensor(weights.vectors)

        self.embedding = nn.Embedding.from_pretrained(weight_vectors, padding_idx=0)
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (window_size, EMBEDDING_SIZE), padding=(window_size - 1, 0))
            for window_size in window_sizes])
        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)
        self.dropout = nn.Dropout(.5)


    def forward(self, x):
        '''
        Implement the forward function to feed the input through the model and get the output.
        1. Feed the input through the embedding layer.
        2. Feed the result through the convolution layers. For convolution layers, pass the convolution output through
        tanh and max_pool1d.
        3. Feed the linear layer output to softmax.

        NOTE : You should use maxpool and softmax functions as well.

        Args:
            inputs: Input data.

        Returns:
            output: Probabilities of each label.
        '''
        '''
        print(x.long().shape)
        embed = self.embedding(x.long())
        print(embed.shape)
        for operation in self.module_list:
            print(operation)
            embed = operation(embed)

        output = nn.Softmax(embed)
        return output
        '''
        embedded = self.embedding(x.long()).unsqueeze(1)
        conv_outputs = [F.tanh(conv(embedded)).squeeze(3) for conv in self.conv_layers]
        pooled_outputs = [F.max_pool1d(output, output.size(2)).squeeze(2) for output in conv_outputs]
        concat = torch.cat(pooled_outputs, dim=1)
        output = self.fc(concat)
        #output = self.dropout(output)
        output = F.softmax(output, dim=1)

        return output