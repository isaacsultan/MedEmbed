import gensim


class Embedding(object):
    def __init__(self, args):
        self.dim = args.dim
        self.model = args.model
        self.workers = args.workers
        self.verbose = args.verbose

    def generate(self, corpus):
        if self.verbose:
            print('Generating word embedding vector with {} model'.format(self.model))

        model = None
        if self.model == 'word2vec':
            model = gensim.models.Word2Vec(corpus, size=self.dim, window=5, workers=self.workers)  # mincount
        elif self.model == 'fasttext':
            model = gensim.models.FastText(corpus, size=self.dim, window=5, workers=self.workers)  # mincount
        filename_model = 'model.bin'
        model.save(filename_model)

        if self.verbose:
            print('Model saved as {}'.format(filename_model))
