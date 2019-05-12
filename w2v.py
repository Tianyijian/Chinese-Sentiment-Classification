import numpy as np
import os
from tensorflow.contrib import learn


def expand_vectors(word_vecs):
    path = "./data/word2vec/merge_sgns_bigram_char300.txt"
    lines_num, dim, origin_line = 0, 0, 0
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                origin_line = int(line.rstrip().split()[0])
                continue
            tokens = line.rstrip().split(' ')
            if tokens[0] not in word_vecs:
                word_vecs[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                lines_num += 1
            if len(word_vecs) >= 50000:
                print("Expand word2vec:  origin_word: {}, expand_word:{}".format(origin_line, lines_num))
                break
    return word_vecs


def expand_vocab(text_list, vocab_processor):
    path = "./data/word2vec/w2v_bigram_char_300.txt"
    vocab = vocab_processor.vocabulary_._reverse_mapping
    origin_vocab_size = len(vocab)
    word_vecs = read_vectors(path)
    word_list = []
    for word in word_vecs.keys():
        if word not in vocab:
            word_list.append(word)
    temp_list = word_list + text_list
    # vocab_processor cannot fit twice
    vocab_processor = learn.preprocessing.VocabularyProcessor(300)
    vocab_processor.fit(temp_list)
    expanded_vocab_size = len(vocab_processor.vocabulary_)
    print("Expand vocab:  origin_vocab_size: {}, expanded_vocab_size: {}".format(origin_vocab_size, expanded_vocab_size))
    return vocab_processor


def read_vectors(path):
    lines_num, dim = 0, 0
    vectors = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
    return vectors


def load_vectors(path, vocab):
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            if tokens[0] in vocab:
                vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                iw.append(tokens[0])
    for i, w in enumerate(iw):
        wi[w] = i
    print("vab size:{}, w2v size: {}, has w2v:{} ".format(len(vocab), lines_num, len(vectors)))
    return vectors


def write_w2v_to_file(path, word_vecs):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("{} {}\n".format(len(word_vecs), 300))
        for word in word_vecs.keys():
            line = word + ' ' + ' '.join(map(str, word_vecs[word].tolist()))
            f.write(line + '\n')


def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-1.0, 1.0, k)


def get_W(word_vecs, vocab, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(vocab)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k), dtype='float32')
    i = 0
    for word in vocab:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    # W[0] = np.zeros(k, dtype='float32')
    return W, word_idx_map


def get_w2v(path, vocab_processor):
    print("Load pre-trained word2vec from " + path)
    # path = "./data/word2vec/merge_sgns_bigram_char300.txt.w2v"  # w2v size: 26633
    vocab = vocab_processor.vocabulary_._reverse_mapping
    word_vecs = load_vectors(path, vocab)
    add_unknown_words(word_vecs, vocab)
    W, word_idx_map = get_W(word_vecs, vocab)
    return W


if __name__ == '__main__':

    path = "./data/word2vec/merge_sgns_bigram_char300.txt.w2v"  # vab size:33018, w2v size: 26633, has w2v:26633
    vocab_path = os.path.join("./runs/1554849291/checkpoints/", "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    vocab = vocab_processor.vocabulary_._reverse_mapping
    # word_vecs = load_vectors(path, vocab)
    # add_unknown_words(word_vecs, vocab)
    # W, word_idx_map = get_W(word_vecs, vocab)

    word_vecs = load_vectors(path, vocab)
    expand_vectors(word_vecs)
    write_w2v_to_file("./data/word2vec/w2v_bigram_char_300.txt", word_vecs)
