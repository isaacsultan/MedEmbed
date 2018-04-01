import gensim


class Embedding:
    """
    methods to generate and evaluate word embedding vector
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def generate(self, corpus, model, dim, workers):
        """
        Models word embedding vector and saves it to file
        :param corpus: processed dataset to model
        :param model: 'word2vec' or 'fasttext'
        :param dim: dimensions of word emb edding vector
        :param workers: number of workers to parallelise training of word embedding model
        :return: None
        """
        if self.verbose:
            print('Generating word embedding vector with {} model'.format(model))

        model = None
        if model == 'word2vec':
            model = gensim.models.Word2Vec(corpus, size=dim, window=5, workers=workers)  # mincount
        elif model == 'fasttext':
            model = gensim.models.FastText(corpus, size=dim, window=5, workers=workers)  # mincount
        filename_model = 'model.bin'
        model.save(filename_model)

        if self.verbose:
            print('Model saved as {}'.format(filename_model))
