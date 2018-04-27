import argparse
import os

from dataset import TxtDataset, XMLDataset
from embedding import Embedding


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
    # parser.add_argument('--ontology', type=str, help='UMLS ontology for semantic mapping and key', default='oncology')
    parser.add_argument('--apikey', type=str, help='API key to access UMLS ontology', default='oncology')
    parser.add_argument('--categories', type=str, help='categories within samples to keep')
    parser.add_argument('--model', type=str, help='choice of word embedding model', default='word2vec', choices=['word2vec', 'fasttext'])
    parser.add_argument('--workers', type=int, help='number of workers to parallelise training of word embedding model',
                        default=1)
    parser.add_argument('--visualise', help='make tSNE plot of trained word embedding model', action = 'count', default=0)

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
    embedding.generate(args.model, args.dim, args.workers)
    if args.visualise:
        embedding.tSNE(model_file='/Users/isaacsultan/Code/MedEmbed/we_models/word2vec_04-27_05:00.bin')

main()