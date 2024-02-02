import gensim


class LDA:
    def __init__(self):
        """
        Initialize LDA Class
        """
        pass

    def tokenize_words(self, inputs):
        """
        Lowercase, tokenize and de-accent sentences to produce the tokens using simple_preprocess function from gensim.utils.

        Args:
            inputs: Input data.

        Returns:
            output: Tokenized list of sentences.
        """

        output = [gensim.utils.simple_preprocess(doc, deacc=False) for doc in inputs]

        return output

    def remove_stopwords(self, inputs, stop_words):
        """
        Remove stopwords from tokenized words.

        Args:
            inputs: Input data.
            stop_words: List of stop_words

        Returns:
            output: Tokenized list of sentences.
        """

        output = [[word for word in sentence if word not in stop_words] for sentence in inputs]

        return output

    def create_dictionary(self, inputs):
        """
        Create dicitionary and term document frequency for the input data using Dicitonary class of gensim.corpora.

        Args:
            inputs: Input data.

        Returns:
            id2word: Gensim index to word map.
            corpus: Term document frequency for each word.
        """

        id2word = gensim.corpora.Dictionary(inputs)
        corpus = [id2word.doc2bow(sen) for sen in inputs]

        return id2word, corpus

    def build_LDAModel(self, id2word, corpus, num_topics=10):
        """
        Build LDA Model using LdaMulticore class of gensim.models.

        Args:
            id2word: Index to word map.
            corpus: Term document frequency for each word.
            num_topics: Number of topics for modeling

        Returns:
            lda_model: LdaMulticore instance.
        """

        lda_model = gensim.models.LdaMulticore(corpus= corpus, num_topics=num_topics, id2word=id2word)

        return lda_model
