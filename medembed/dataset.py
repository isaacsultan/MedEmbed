import fileinput
import os

import gensim
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


class DataSet(object):
    def __init__(self, args):
        self.args = args
        self.dir_processed = os.path.join(os.path.dirname(args.dir), 'processed')
        self.clevermap = self._make_clever_map()
        self.categories = self._make_categories()
        self.bigrams = None
        self.word_freqs = dict()
        self.dictionary = None


    @staticmethod
    def iter_documents(dir):
        """
        Generator: iterate over all relevant documents, yielding one
        document (=list of utf8 tokens) at a time.
        """
        for root, dirs, files in os.walk(dir):
            for fname in filter(lambda fname: fname.endswith('.txt'), files):
                document = open(os.path.join(root, fname)).read()
                yield gensim.utils.tokenize(document, errors='ignore')

    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in DataSet.iter_documents():
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)

    def preprocess(self):
        if self.args.verbose:
            print('Processing files in directory {}'.format(self.args.dir))
        self._read_extract()

        if self.args.verbose:
            print('Finding word and bigram frequencies')
        self.bigrams = self._find_frequent(500)
        if self.args.verbose:
            print('{} unique words after processing'.format(len(self.args.word_freqs)))

        self.dictionary = gensim.corpora.Dictionary(DataSet.iter_documents())

        if self.args.verbose:
            print('Performing UMLS and CLEVER mapping')
        self._do_mapping()

    def _read_extract(self):
        stemmer = SnowballStemmer('english')
        stops = set(stopwords.words('english'))
        category = set(self.categories)
        category_tokens = set([' '.split(i) for i in self.categories])
        directory_files = os.listdir(self.args.dir)

        if self.args.verbose:
            print('{} files found'.format(len(directory_files)))

        file_count = 0
        for file in directory_files:
            if file.endswith(".txt"):
                with open(os.path.join(self.args.dir, file), 'r') as f:
                    clean_sample = self._make_clean_sample(f, stops, stemmer, category, category_tokens)
                with open(os.path.join(self.dir_processed, file), 'w') as f:
                    f.write(str(clean_sample))
            if self.args.verbose and file_count % 500:
                print('Processed {} files'.format(file_count))

    def _make_categories(self):
        if self.args.categories is None:
            return None
        else:
            with open(self.args.categories, 'r') as f:
                return f.readlines()

    def _make_clean_sample(self, f, stops, stemmer, category, category_tokens):
        clean_sample = []

        for line in f:
            if any(s in line for s in category):
                tokens = word_tokenize(line)
                for token in tokens:
                    if token not in stops and token not in category_tokens and token.isalpha():
                        if token not in self.word_freqs:
                            self.word_freqs[token] = 0
                        else:
                            self.word_freqs[token] += 1
                        token = stemmer.stem(token)
                        token = token.lower()
                        clean_sample.append(token)
                clean_sample.append('\n')
        return clean_sample

    def _find_frequent(self, threshold_bigram):
        bigram_dict = dict()
        with fileinput.input(files=os.listdir(self.dir_processed)) as f:
            for line in f:
                for i in range(len(line) - 1):
                    if line[i] != '\n' and line[i + 1] != '\n':  # check
                        bigram = line[i] + '_' + line[i + 1]
                        if bigram not in bigram_dict:
                            bigram_dict[bigram] = 0
                        else:
                            bigram_dict[bigram] += 1
        bigram_dict['renal_cell'] = 600  # TEST REMOVE LATER
        sorted_bigrams = set([k for (k, v) in bigram_dict.items() if v > threshold_bigram])
        return sorted_bigrams

    def _encode_negation(self):
        return None

    def _make_clever_map(self):
        clever_map = dict()
        with open('/Users/isaacsultan/hig/isaac/clever_term/clever_base_terminology.txt', 'r') as f:
            for line in f:
                tokens = line.split('|')
                clever_map[tokens[1]] = tokens[2].strip('\n')
        return clever_map

    def _make_umls_map(self):
        return None

    def _do_mapping(self):
        for line in fileinput.input(files=os.listdir(self.dir_processed), inplace=True):
            tokens = word_tokenize(line)
            newline = ' '.join(str(self.clevermap.get(token, token)) for token in tokens)
            print(newline)  # check