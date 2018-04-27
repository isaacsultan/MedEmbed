import fileinput
import os

from nltk.tokenize import word_tokenize

from . import DIR_PROCESSED


class Transformer:
    """
    Methods that process the dataset before generating word embedding model
    """

    def __init__(self, categories, apikey=None):

        self.category = None
        self.category_tokens = set([' '.split(i) for i in categories])
        self.word_freqs = dict()
        self.apikey = apikey

    def make_clean_sample(self, f, stops, stemmer, ftype='xml'):
        """
        raw text -> clean text and generates word_frequency dictionary
        :param f: raw text
        :param stops: set of stopwords
        :param stemmer: nltk stemmer
        :param ftype: type of files to process
        :return: processed text
        """
        clean_sample = ''

        if ftype == 'xml':
            tokens = word_tokenize(f)
            for token in tokens:
                if token in ['?', '!', '.']:
                    clean_sample += '\n'
                elif token not in stops and token.isalpha():
                    if token not in self.word_freqs:
                        self.word_freqs[token] = 0
                    else:
                        self.word_freqs[token] += 1
                    token = stemmer.lemmatize(token)
                    token = token.lower()
                    clean_sample += token + ' '
            return clean_sample

        for line in f:
            if self.category is None:
                clean_line = ''
                tokens = word_tokenize(line)
                for token in tokens:
                    if token not in stops and token.isalpha():
                        if token not in self.word_freqs:
                            self.word_freqs[token] = 0
                        else:
                            self.word_freqs[token] += 1
                        token = stemmer.lemmatize(token)
                        token = token.lower()
                        clean_line += token + ' '
                clean_sample += clean_line

            else:
                if any(s in line for s in self.category):
                    clean_line = ''
                    tokens = word_tokenize(line)
                    for token in tokens:
                        if token not in stops and token not in self.category_tokens and token.isalpha():
                            if token not in self.word_freqs:
                                self.word_freqs[token] = 0
                            else:
                                self.word_freqs[token] += 1

                        token = stemmer.lemmatize(token)
                        token = token.lower()
                        clean_line += token + ' '
                    clean_sample += clean_line
        return clean_sample

    def transform(self):
        clever_map = self._make_clever_map()
        # umls_map = self._make_umls_map()
        self._do_mapping(clever_map)

    @staticmethod
    def _make_clever_map():
        clever_map = dict()
        fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'clever_term', 'clever_base_terminology.txt')
        with open(fname, 'r') as f:
            for line in f:
                tokens = line.split('|')
                clever_map[tokens[1]] = tokens[2].strip('\n')
        return clever_map

    def _make_umls_map(self):
        return None

    def _do_mapping(self, clever_map, umls_map=None):

        files = os.listdir(DIR_PROCESSED)
        for fname in filter(lambda fname: fname.endswith('.txt'), files):
            full_fname = os.path.join(DIR_PROCESSED, fname)
            with open(full_fname, 'r') as f:
                mapped_text = ''
                for line in f:
                    tokens = word_tokenize(line)
                    newline = ' '.join(str(clever_map.get(token, token)) for token in tokens) + '\n'
                    mapped_text += newline
                print(mapped_text, file=open(full_fname, 'w'))

    @staticmethod
    def _find_frequent(threshold_bigram):
        """
        Reads preprocessed files and counts bigrams
        :param threshold_bigram: minimum frequency of bigram
        :return: all bigrams that occur more than threshold_bigram times
        """
        bigram_dict = dict()
        with fileinput.input(files=os.listdir(DIR_PROCESSED)) as f:
            for line in f:
                for i in range(len(line) - 1):
                    if line[i] != '\n' and line[i + 1] != '\n':
                        bigram = line[i] + '_' + line[i + 1]
                        if bigram not in bigram_dict:
                            bigram_dict[bigram] = 0
                        else:
                            bigram_dict[bigram] += 1
        sorted_bigrams = set([k for (k, v) in bigram_dict.items() if v > threshold_bigram])
        return sorted_bigrams
