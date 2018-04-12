import argparse
import os

from dataset import TxtDataset, XMLDataset
from embedding import Embedding

DIR_PROCESSED = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed')

def get_arguments():
    """
    Defines and reads command line arguments
    :return: Command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate word embedding vectors')
    parser.add_argument('-v', '--verbose', help='verbose mode', action='count', default=0)
    parser.add_argument('dir', type=str,
                        help='directory to read files')
    parser.add_argument('--filetype', type=str, help='raw data filetype', default='txt', choices=['txt', 'xml'])
    parser.add_argument('--dim', type=int, help='dimensions of word embedding vectors', default=200)
    #parser.add_argument('--ontology', type=str, help='UMLS ontology for semantic mapping and key', default='oncology')
    parser.add_argument('--apikey', type=str, help='API key to access UMLS ontology', default='oncology')
    parser.add_argument('--categories', type=str, help='categories within samples to keep')
    parser.add_argument('--model', type=str, default='word2vec', choices=['word2vec', 'fasttext'])
    parser.add_argument('--workers', type=int, help='number of workers to parallelise training of word embedding model',
                        default=1)

    return parser.parse_args()


def main():
    """
    dataset -> transformed dataset -> word embedding vector
    :return: None
    """

    args = get_arguments()

    if not os.listdir(args.dir):
        raise ValueError('No files found in file directory')


    if args.filetype == 'txt':
        dataset = TxtDataset(args.dir, args.verbose, args.categories)
    else:
        dataset = XMLDataset(args.dir, args.verbose, args.categories)

    dataset.preprocess()
    embedding = Embedding(args.verbose)
    embedding.generate(dataset, args.model, args.dim, args.workers)


if '__name__ == __main__':
    main()
