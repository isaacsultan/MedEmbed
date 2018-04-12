import gensim
import datetime


class Embedding:
    """
    methods to generate and evaluate word embedding vector
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def generate(self, corpus, model_type, dim, workers):
        """
        Models word embedding vector and saves it to file
        :param corpus: processed dataset to model
        :param model_type: 'word2vec' or 'fasttext'
        :param dim: dimensions of word emb edding vector
        :param workers: number of workers to parallelise training of word embedding model
        :return: None
        """
        if self.verbose:
            print('Generating word embedding vector with {} model_type'.format(model_type))


        for i in corpus:
            print(i)

        model = None
        if model_type == 'word2vec':
            model = gensim.models.Word2Vec(corpus, size=dim, window=5, workers=workers)  # mincount
            print('done')
        elif model_type == 'fasttext':
            model = gensim.models.FastText(corpus, size=dim, window=5, workers=workers)  # mincount

        now = datetime.datetime.now()

        filename_model = model_type + now.strftime('%Y-%m-%d') + '.bin'
        model.save(filename_model)

        if self.verbose:
            print('Model saved as {}'.format(filename_model))
