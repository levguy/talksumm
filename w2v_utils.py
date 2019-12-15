import pandas as pd
import numpy as np
from util import tprint
from gensim.models import KeyedVectors


def read_pretrained_w2v(pretrained_w2v_fname, is_glove):
    """
    reads pretrained word embedding from the given file name
    is_glove: if True, assumes the format of GloVe text file. otherwise, word2vec bin file is assumed.
    in Glove case: returns a dictionary which maps a word to its vector
    in word2vec case: returns Word2VecKeyedVectors (of gensim)
    in addition, mean_vec is returned, which is the mean of all vectors (can be used for <unk>)
    """
    tprint("reading file: {}".format(pretrained_w2v_fname))

    if is_glove:
        w2vec = pd.read_csv(pretrained_w2v_fname, header=None, sep=' ', quoting=3, encoding="ISO-8859-1")
        tprint("done")

        w2v_words = w2vec.iloc[:, 0].values
        w2v_vectors = w2vec.iloc[:, 1:].values

        num_words, dim = w2v_vectors.shape

        mean_vec = np.mean(w2v_vectors, 0)

        w2v = {}

        for word_i, word in enumerate(w2v_words):
            w2v[word] = w2v_vectors[word_i, :]

    else:
        w2v = KeyedVectors.load_word2vec_format(pretrained_w2v_fname, binary=True)
        tprint("done")

        num_words = len(w2v.vocab)
        dim = w2v.vector_size

        mean_vec = np.mean(w2v.syn0, 0)

    print("dim: {}".format(dim))
    print("num_words: {}".format(num_words))

    return w2v, mean_vec
