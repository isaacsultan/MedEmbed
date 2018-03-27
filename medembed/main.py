import argparse

from dataset import DataSet
from embedding import Embedding


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate word embedding vectors')
    parser.add_argument('-v', '--verbose', help='verbose mode', action='count', default=0)
    parser.add_argument('dir', type=str,
                        help='directory to read files')
    parser.add_argument('--filetype', type=str, help='raw data filetype', default='txt', choices=['txt', 'xml'])
    parser.add_argument('--dim', type=int, help='dimensions of word embedding vectors', default=200)
    parser.add_argument('--ontology', type=str, help='UMLS ontology for semantic mapping and key', default='oncology')
    parser.add_argument('--apikey', type=str, help='API key to access UMLS ontology', default='oncology')
    parser.add_argument('--categories', type=str, help='categories within samples to keep')
    parser.add_argument('--model', type=str, default='glove', choices=['glove', 'fasttext'])
    parser.add_argument('--workers', type=int, help='number of workers to parallelise training of word embedding model',
                        default=1)

    return parser.parse_args()


def main():

    args = get_arguments()
    # authentification = Authentication(args.apikey)
    dataset = DataSet(args)
    dataset.preprocess()
    #embedding = Embedding(args)
    #embedding.generate(dataset)


if '__name__ == __main__':
    main()
