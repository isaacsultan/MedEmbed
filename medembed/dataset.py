import os

from transformer import Transformer

try:
    # noinspection PyPep8Naming
    import xml.etree.cElementTree as ET
except ImportError:
    # noinspection PyPep8Naming
    import xml.etree.ElementTree as ET

import gensim
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

DIR_PROCESSED = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'processed')


class DataSet:
    """
    Holds the dataset and the methods associated with it
    """

    def __init__(self, dir, verbose):
        self.dir = dir
        self.verbose = verbose
        self.dictionary = None

    def preprocess(self):
        """
        Calls pre-processing methods and prints progress (if verbose)
        :return: None
        """

        if self.verbose:
            print('Processing files in directory {}'.format(self.dir))

        categories = self._make_categories()  # TODO:refactor
        transformer = Transformer(categories, self.args.apikey)
        self._read_extract(transformer)

        if self.verbose:
            print('Finding word and bigram frequencies')

        if self.verbose:
            print('{} unique words after processing'.format(len(transformer.word_freqs)))

        self.dictionary = gensim.corpora.Dictionary(DataSet.iter_documents())

        if self.verbose:
            print('Performing UMLS and CLEVER mapping')
        transformer.transform()

    @staticmethod
    def iter_documents():
        """
        Generator: iterate over all relevant documents
        :return: yields one document (=list of utf8 tokens) at a time
        """
        for root, dirs, files in os.walk(DIR_PROCESSED):
            for fname in filter(lambda fname: fname.endswith('.txt'), files):
                document = open(os.path.join(root, fname)).read()
                yield gensim.utils.tokenize(document, errors='ignore')

    def __iter__(self):
        """
        __iter__ is a generator => Dataset is a streamed iterable
        :return: sparse dictionary
        """
        for tokens in DataSet.iter_documents():
            yield self.dictionary.doc2bow(tokens)


class XMLDataset(DataSet):

    def _read_extract(self, transformer):
        """
        Reads xml files in data directory, cleans files and writes each file to preprocessed directory
        :return: None
        """

        stemmer = SnowballStemmer('english')
        stops = set(stopwords.words('english'))

        directory_files = os.listdir(self.dir)

        if self.verbose:
            print('{} files found'.format(len(directory_files)))

        file_count = 0

        for fname in filter(lambda fname: fname.endswith('.xml'), directory_files):
            with open(os.path.join(self.dir, fname), 'r') as f:
                clean_sample = transformer.make_clean_sample(f, stops, stemmer)
            new_fname = os.path.join(DIR_PROCESSED, fname)
            print(clean_sample, file=new_fname)
            file_count += 1

            if self.verbose and file_count % 500:
                print('Processed {} files'.format(file_count))


class TxtDataset(DataSet):

    def _read_extract(self, transformer):
        """
        Reads txt files in data directory, cleans files and writes each file to preprocessed directory
        :return: None
        """

        stemmer = SnowballStemmer('english')
        stops = set(stopwords.words('english'))

        directory_files = os.listdir(self.dir)

        if self.verbose:
            print('{} files found'.format(len(directory_files)))

        file_count = 0

        for fname in filter(lambda fname: fname.endswith('.xml'), directory_files):
            with open(os.path.join(self.dir, fname), 'r') as f:
                clean_sample = transformer.make_clean_sample(f, stops, stemmer)
            new_fname = os.path.join(DIR_PROCESSED, fname)
            print(clean_sample, file=new_fname)
            file_count += 1

            if self.verbose and file_count % 500:
                print('Processed {} files'.format(file_count))


    def _make_categories(self):
        """
        Makes a list of categories to extract from a raw document
        :return: category list, or None (if extracting all categories)
        """

        if self.args.categories is not None:
            with open(self.args.categories, 'r') as f:
                return f.readlines()
