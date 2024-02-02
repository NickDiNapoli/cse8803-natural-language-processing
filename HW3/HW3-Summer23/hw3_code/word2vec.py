import numpy as np
import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn


class Word2Vec(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocabulary_size = 0

    def tokenize(self, data):
        """
        Split all the words in the data into tokens.

        Args:
            data: (N,) list of sentences in the dataset.

        Return:
            tokens: (N, D_i) list of tokenized sentences. The i-th sentence has D_i words after the split.
        """
        data_split = [sentence.split() for sentence in data]
        # print(data_split)
        # tokens = [[word.lower() for word in sen] for sen in data_split]
        tokens = data_split
        return tokens


    def create_vocabulary(self, tokenized_data):
        """
        Create a vocabulary for the tokenized data.
        For each unique word in the vocabulary, assign a unique ID to the word. Please sort the vocabulary before assigning index.

        Assign the word to ID mapping to word2idx variable.
        Assign the ID to word mapping to idx2word variable.
        Assign the size of the vocabulary to vocabulary_size variable.

        Args:
            tokenized_data: (N, D) list of split tokens in each sentence.
            
        Return:
            None (The update is done for self.word2idx, self.idx2word and self.vocabulary_size)
        """
        vocab = sorted(set(word for sen in tokenized_data for word in sen))
        for i, w in enumerate(vocab):
            self.word2idx[w] = i
            self.idx2word[i] = w

        self.vocabulary_size = len(self.word2idx)


    def skipgram_embeddings(self, tokenized_data, window_size=2):
        """
        Create a skipgram embeddings by taking context as middle word and predicting
        N=window_size past words and N=window_size future words.

        NOTE : In case the window range is out of the sentence length, create a
        context by feasible window size. For example : The sentence with tokenIds
        as follows will be represented as
        [1, 2, 3, 4, 5] ->
           source_tokens             target_tokens
           [1]                       [2]
           [1]                       [3]
           [2]                       [1]
           [2]                       [3]
           [2]                       [4]
           [3]                       [1]
           [3]                       [2]
           [3]                       [4]
           [3]                       [5]
           [4]                       [2]
           [4]                       [3]
           [4]                       [5]
           [5]                       [3]
           [5]                       [4]

        source_tokens: [[1], [1], [2], [2], [2], ...]
        target_tokens: [[2], [3], [1], [3], [4], ...]
        Args:
            tokenized_data: (N, D_i) list of split tokens in each sentence.
            window_size: length of the window for creating context. Default is 2.

        Returns:
            source_tokens: List of elements where each element is the middle word in the window.
            target_tokens: List of elements representing IDs of the context words.
        """
        source_tokens, target_tokens = [], []
        for sentence in tokenized_data:
            source_tok, target_tok = [], []
            for token_idx in range(len(sentence)):

                start_idx = max(0, token_idx - window_size)
                end_idx = min(len(sentence), token_idx + window_size + 1)

                for idx in range(start_idx, end_idx):
                    if idx != token_idx:
                        target_tokens.append([self.word2idx[sentence[idx]]])
                        source_tokens.append([self.word2idx[sentence[token_idx]]])

                # target_tokens.append([sentence[token_idx]])
                # source_tokens.append(sor_tok)
            target_tokens.append(target_tok)
            source_tokens.append(source_tok)

        source_tokens = [x for x in source_tokens if len(x) > 0]
        target_tokens = [x for x in target_tokens if len(x) > 0]
        return source_tokens, target_tokens
    

    def cbow_embeddings(self, tokenized_data, window_size=2):
        """
        Create a cbow embeddings by taking context as N=window_size past words and N=window_size future words.

        NOTE : In case the window range is out of the sentence length, create a
        context by feasible window size. For example : The sentence with tokenIds
        as follows will be represented as
        [1, 2, 3, 4, 5] ->
           source_tokens             target_tokens
           [2,3]                     [1]
           [1,3,4]                   [2]
           [1,2,4,5]                 [3]
           [2,3,5]                   [4]
           [3,4]                     [5]
           
        source_tokens: [[2,3], [1,3,4], [1,2,4,5], [2,3,5], [3,4]]
        target_tokens: [[1], [2], [3], [4], [5]]

        Args:
            tokenized_data: (N, D_i) list of split tokens in each sentence.
            window_size: length of the window for creating context. Default is 2.

        Returns:
            source_tokens: List of elements where each element is maximum of N=window_size*2 context word IDs.
            target_tokens: List of elements representing IDs of the middle word in the window.
        """
        # the source and target tokens are lists of lists of lists
        # where the outer list is a sentence
        # example sentence -> [1, 2, 3, 4, 5]
        '''
        source_tokens, target_tokens = [], []
        for sentence in tokenized_data:
            source_tok, target_tok = [], []
            for token_idx in range(len(sentence)):
                target_tok.append([self.word2idx[sentence[token_idx]]])

                start_idx = max(0, token_idx - window_size)
                end_idx = min(len(sentence), token_idx + window_size + 1)

                sor_tok = []
                for idx in range(start_idx, end_idx):
                    if idx != token_idx:
                        sor_tok.append(self.word2idx[sentence[idx]])

                source_tok.append(sor_tok)

                #target_tokens.append([sentence[token_idx]])
                #source_tokens.append(sor_tok)
            target_tokens.append(target_tok)
            source_tokens.append(source_tok)
            
        return source_tokens, target_tokens
        '''
        source_tokens = []
        target_tokens = []

        for sentence in tokenized_data:
            sentence_length = len(sentence)

            for idx in range(sentence_length):
                context = []
                target = self.word2idx[sentence[idx]]

                start_index = max(0, idx - window_size)
                end_index = min(sentence_length, idx + window_size + 1)

                for j in range(start_index, end_index):
                    if j != idx:
                        context.append(self.word2idx[sentence[j]])

                source_tokens.append(context)
                target_tokens.append([target])

        return source_tokens, target_tokens


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initialize SkipGram_Model with the embedding layer and a linear layer.
        Please define your embedding layer before your linear layer.
        
        Reference: 
            embedding - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            linear layer - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        Args:
            vocab_size: Size of the vocabulary.
        """
        super(SkipGram_Model, self).__init__()
        self.EMBED_DIMENSION = 300 # please use this to set embedding_dim in embedding layer
        self.EMBED_MAX_NORM = 1    # please use this to set max_norm in embedding layer

        self.embedding = nn.Embedding(vocab_size, self.EMBED_DIMENSION, max_norm=self.EMBED_MAX_NORM)
        self.linear_layer = nn.Linear(in_features=self.EMBED_DIMENSION, out_features=vocab_size)  # ,bias=False)


    def forward(self, inputs):
        """
        Implement the SkipGram model architecture as described in the notebook.

        Args:
            inputs: Tensor of IDs for each sentence.

        Returns:
            output: Tensor of logits with shape same as vocab_size.
            
        Hint:
            No need to have a softmax layer here.
        """
        embedding = self.embedding(inputs)
        #print(embedding.shape)
        #flattened = embedding.view(embedding.size(0), -1)
        #print(flattened.shape)
        #embedding = embedding.squeeze()
        #print(embedding.shape)
        output = self.linear_layer(embedding).squeeze()
        #print(output.shape)

        return output


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initialize CBOW_Model with the embedding layer and a linear layer.
        Please define your embedding layer before your linear layer.
        
        Reference: 
            embedding - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            linear layer - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        Args:
            vocab_size: Size of the vocabulary.
        """
        super(CBOW_Model, self).__init__()
        self.EMBED_DIMENSION = 300  # please use this to set embedding_dim in embedding layer
        self.EMBED_MAX_NORM = 1     # please use this to set max_norm in embedding layer

        self.embedding = nn.Embedding(vocab_size, self.EMBED_DIMENSION, max_norm=self.EMBED_MAX_NORM)
        self.linear_layer = nn.Linear(in_features=self.EMBED_DIMENSION, out_features=vocab_size) #, bias=False)

    def forward(self, inputs):
        """
        Implement the CBOW model architecture as described in the notebook.

        Args:
            inputs: Tensor of IDs for each sentence.

        Returns:
            output: Tensor of logits with shape same as vocab_size.
            
        Hint:
            No need to have a softmax layer here.
        """
        #print(inputs.shape)
        embedding = self.embedding(inputs)
        #print(embedding.shape)
        averaging = torch.mean(embedding, dim=0)
        #print(averaging.shape)
        averaging = averaging.unsqueeze(0)
        output = self.linear_layer(averaging)
        #output = torch.argmax(torch.softmax(output, dim=-1)).unsqueeze(0)
        #print(output.shape)

        return output.float()
