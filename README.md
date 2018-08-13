# MedEmbed

An end-to-end tool to train relevent word embedding models on medical clinical texts.

## Project Motivation

Word embeddings are arguably the most widely known best practice in the recent history of NLP.

No pre-trained word embedding models exist for *Radiological Oncology*, and many other medical specialties.
Generating word embeddings on raw clinical texts, present unique challenges on account of a lack of consistant formatting
and the prevelence of specialist vocabulary.

## Installation

    git clone https://github.com/isaacsultan/MedEmbed
    cd MedEmbed
    pip install -r requirements.txt


### Requirements

* python 3
* pip
* git

## Usage

    usage: main.py
                [-h] # help
                [-v] # verbosity
                [--filetype {txt,xml}] # type of input file
                [--dim DIM]  # dimensions of WE vector generated
                [--ontology ONTOLOGY] # UMLS ontology for semantic mapping (requires --apikey)
                [--apikey APIKEY] # API key to access UMLS ontology
                [--categories CATEGORIES] # text file with headers of categories to parse
                [--model {glove,fasttext}] # WE model to generate
                [--workers WORKERS] # train WE model over n cores
                ['--visualise] # make tSNE plot of trained word embedding model
                dir # directory of input files (absolute path)

### Documentation

[Read the Docs](http://medembed.readthedocs.io)

Examples within popular deep learning frameworks of WE use for text classification.

1. TensorFlow:  [tensorflow_fasttext](https://github.com/apcode/tensorflow_fasttext)
2. Pytorch: [DeepLearningForNLPInPytorch](https://github.com/rguthrie3/DeepLearningForNLPInPytorch)
3. Keras: [keras_gensim_embeddings](https://gist.github.com/codekansas/15b3c2a2e9bc7a3c345138a32e029969)

## Authors

* **Isaac Sultan** - *Initial work* - [Github](https://github.com/isaacsultan)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Radiology report annotation using intelligent word embeddings: [Banerjee et al., 2017](https://doi.org/10.1016/j.jbi.2017.11.012)
