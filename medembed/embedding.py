import datetime
import os
import random

import gensim
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from . import DIR_PROCESSED


class Embedding:
    """
    methods to generate and evaluate word embedding vector
    """

    def __init__(self, verbose):
        self.verbose = verbose
        self.trained_model = None
        self.name_model = None

    def generate(self, model_type, dim, workers):
        """
        Models word embedding vector and saves it to file
        :param model_type: 'word2vec' or 'fasttext'
        :param dim: dimensions of word emb edding vector
        :param workers: number of workers to parallelise training of word embedding model
        :return: None
        """
        if self.verbose:
            print('Generating word embedding vector with {} model_type'.format(model_type))

        model = None
        if model_type == 'word2vec':
            sentences = gensim.models.word2vec.PathLineSentences(DIR_PROCESSED)
            model = gensim.models.word2vec.Word2Vec(sentences, size=dim, window=5, sg=1, workers=workers)
        elif model_type == 'fasttext':
            sentences = gensim.models.word2vec.PathLineSentences(DIR_PROCESSED)
            model = gensim.models.FastText(sentences, size=dim, window=5, workers=workers)

        now = datetime.datetime.now()

        name_model = model_type + '_' + now.strftime('%m-%d_%H:%M')
        fname_model = name_model + '.bin'
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'we_models')
        full_fname = os.path.join(save_path, fname_model)

        if not os.path.exists(save_path):
            if self.verbose:
                print('Creating a directory for processed files at {}'.format(save_path))
            os.makedirs(save_path)

        model.save(full_fname)

        if self.verbose:
            print('Model saved as {} in Medembed/we_models'.format(fname_model))

        self.trained_model = model
        self.name_model = name_model


    def tSNE(self, model_file = None):
        """
        Creates TSNE model, plots it and saves it
        :return: None
        """

        if self.verbose:
            print('Generating tSNE plot')

        if model_file is not None:
            model = gensim.models.KeyedVectors.load(model_file)
            self.name_model = model_file.replace('.bin', '')
        else:
            model = self.trained_model

        labels = []
        tokens = []

        for word in model.wv.vocab:
            tokens.append(model[word])
            if random.random() < 0.05:
                labels.append(word)
            else:
                labels.append(' ')

        tsne_model = TSNE(perplexity=30, n_components=2, init='pca', random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))
        sns.set()
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        save_path =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
        if not os.path.exists(save_path):
            if self.verbose:
                print('Creating a directory for plot files at {}'.format(save_path))
            os.makedirs(save_path)

        fname = self.name_model +'.png'
        plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')

        if self.verbose:
            print('Plot saved as {} in Medembed/plots'.format(fname))
