import os

from .transformer import Transformer

try:
    # noinspection PyPep8Naming
    import xml.etree.cElementTree as ET
except ImportError:
    # noinspection PyPep8Naming
    import xml.etree.ElementTree as ET

import gensim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from medembed import DIR_PROCESSED


class DataSet:
    """
    Holds the dataset and the methods associated with it
    """

    def __init__(self, directory, verbose, categories):
        self.dir = directory
        self.verbose = verbose
        self.dictionary = None
        self.categories = categories
        self.type = None

        self.stemmer = WordNetLemmatizer()
        self.stops = self._make_stops()

    def _read_extract(self, transformer):
        raise NotImplementedError

    @staticmethod
    def _make_stops():
        stops = set(stopwords.words('english'))
        stops.difference_update({'no', 'nor', 'not'})
        return stops

    def preprocess(self):
        """
        Calls pre-processing methods and prints progress (if verbose)
        :return: None
        """

        if self.verbose:
            print('Processing files in directory {}'.format(self.dir))

        categories = self._make_categories(self.categories)
        transformer = Transformer(categories)

        if not os.path.exists(DIR_PROCESSED):
            if self.verbose:
                print('Creating a directory for processed files at {}'.format(DIR_PROCESSED))
            os.makedirs(DIR_PROCESSED)

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

    @staticmethod
    def _make_categories(categories):
        """
        Makes a list of categories to extract from a raw document
        :return: category list, or None (if extracting all categories)
        """

        if categories is None:
            return []
        else:
            with open(categories, 'r') as f:
                return f.readlines()


class XMLDataset(DataSet):
    def __init__(self, directory, verbose, categories):
        super().__init__(directory, verbose, categories)
        self.type = 'xml'

    def _read_extract(self, transformer):
        """
        Reads xml files in data directory, cleans files and writes each file to preprocessed directory
        :return: None
        """

        directory_files = os.listdir(self.dir)

        if self.verbose:
            print('{} files found'.format(len(directory_files)))

        file_count = 0
        for fname in filter(lambda fname: fname.endswith('.xml'), directory_files):
            tree = ET.parse(os.path.join(self.dir, fname))
            f = ET.tostring(tree.getroot()[0]).decode()
            clean_sample = transformer.make_clean_sample(f, self.stops, self.stemmer, self.type)
            new_fname = fname.split('.xml')[0] + '.txt'
            new_fname = os.path.join(DIR_PROCESSED, new_fname)
            print(clean_sample, file=open(new_fname, 'w'))
            file_count += 1

            if self.verbose and file_count % 50 == 0:
                print('Processed {} files'.format(file_count))


class TxtDataset(DataSet):
    def __init__(self, directory, verbose, categories):
        super().__init__(directory, verbose, categories)
        self.type = 'txt'

    def _read_extract(self, transformer):
        """
        Reads txt files in data directory, cleans files and writes each file to preprocessed directory
        :return: None
        """

        directory_files = os.listdir(self.dir)

        if self.verbose:
            print('{} files found'.format(len(directory_files)))

        file_count = 0

        for fname in filter(lambda fname: fname.endswith('.txt'), directory_files):
            with open(os.path.join(self.dir, fname), 'r') as f:
                clean_sample = transformer.make_clean_sample(f, self.stops, self.stemmer, self.type)
            new_fname = os.path.join(DIR_PROCESSED, fname)
            print(clean_sample, file=open(new_fname, 'w'))
            file_count += 1

            if self.verbose and file_count % 50 == 0:
                print('Processed {} files'.format(file_count))
