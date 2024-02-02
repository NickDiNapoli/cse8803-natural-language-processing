import numpy as np
import os
from scipy import spatial


class Glove(object):
    def __init__(self):
        pass

    def load_glove_model(self, glove_file=os.path.join("data/glove.6B.50d.txt")):
        """
        Loads the glove model from the file.
        NOTE: This function is given and does not have to be implemented or changed.

        Args:
                glove_file: file path for the glove model.

        Return:
                glove_model: the embedding for each word in a dict format.
        """
        print("Loading Glove Model")
        glove_model = {}
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        print(f"{len(glove_model)} words loaded!")
        return glove_model

    def find_similar_word(self, model, emmbeddings):
        """
        Finds nearest similar word based on euclidean distance.
        NOTE: This function is given and does not have to be implemented or changed.

        Args:
                model: The glove model in a dict format.
                embeddings: The embeddings for which the nearest similar word needs to be found.

        Return:
                nearest: The nearest similar word for the given embeddings.
        """
        nearest = sorted(
            model.keys(),
            key=lambda word: spatial.distance.euclidean(model[word], emmbeddings),
        )
        return nearest

    def transform(self, model, data, dimension=50):
        """
        Transform the given data to a numpy array of glove features by taking a mean of all the embeddings for each word in a given sentence. Any token which is not present in the glove model should be ignored.
               
        Args:
                model: The glove model in a dict format.
                data: (N,) The list of sentences to be transformed to embeddings.
                dimension: Dimension of the output embeddings.

        Return:
                transformed_features: (N,D) The extracted glove embeddings for each sentence with mean of all words in a sentence.

        Hint:
            You may find try/except block useful to deal with word that does not exist in the model
        """
        #print(data[0].split())
        #print(list(model.keys())[0:10])
        #print(model['the'])
        embeddings = []
        for sentence in data:
            # sen_embed is a list of arrays of shape
            sen_embed = []
            cleaned_sen = [w.lower() for w in sentence.split()]
            #print(cleaned_sen)
            for word in cleaned_sen:
                try:
                    sen_embed.append(model[word])
                    #print('true')
                except KeyError:
                    continue
                    #sen_embed.append(np.zeros(shape=(dimension,)))

                #if sen_embed:
                    #embeddings.append(sen_embed)
            #print(len(sen_embed), sen_embed[0].shape)
            embeddings.append(np.mean(sen_embed, axis=0))
            #embeddings.append(sen_embed)

        #embeddings = np.asarray(embeddings)
        #print(embeddings.shape)
        #transformed_features = np.mean(embeddings, axis=1)
        '''
        transformed_features = []
        for n in range(len(data)):
            transformed_features.append(np.mean(embeddings[n], axis=0))

        transformed_features = np.asarray(transformed_features)
        
        transformed_features = np.zeros((len(embeddings), dimension))
        for i, sen_embed in enumerate(embeddings):
            transformed_features[i] = np.mean(sen_embed, axis=0)
        '''

        transformed_features = np.asarray(embeddings)
        #print(transformed_features.shape)
        return transformed_features
