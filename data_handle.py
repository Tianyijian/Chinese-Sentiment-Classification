from html.parser import HTMLParser
from html.entities import name2codepoint
import codecs
import csv
import numpy as np
import jieba
from tensorflow.contrib import learn
import w2v

positive_train_data = "./data/train_data/sample.positive.txt"
negative_train_data = "./data/train_data/sample.negative.txt"
test_data = "./data/test_data/test.txt"
test_data_with_label = './data/test_data/test.label.cn.xml'
data_list = []
label_list = []


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        # print("Start tag:", tag)
        # for attr in attrs:
        #     print("     attr:", attr)
        if len(attrs) > 1:
            label_list.append(int(attrs[1][1]))

    #
    # def handle_endtag(self, tag):
    #     print("End tag  :", tag)

    def handle_data(self, data):
        # print("Data     :", data.replace('\n', '')[0:50])
        content = data.replace('\n', '').replace('\r', '')
        if not content == '':
            data_list.append(content)


def data_parse():
    """HTMLParser parse the train data"""
    parser = MyHTMLParser()
    parser.feed(codecs.open(positive_train_data, "r", "utf-8").read())
    pos_list = data_list[1:]
    data_list.clear()
    parser.feed(codecs.open(negative_train_data, "r", "utf-8").read())
    neg_list = data_list[1:]
    data_list.clear()
    return pos_list, neg_list


def data_analysis(data_list, tag):
    """data analysis"""
    total_length = sum(len(x) for x in data_list)
    max_length = max(len(x) for x in data_list)
    print("{:s}_data: sent_num: {:d}, max_word_length: {:d}, aver_word_length: {:f}".format(tag, len(data_list),
                                                                                            max_length,
                                                                                            total_length / len(
                                                                                                data_list)))
    count = 0
    for x in data_list:
        if len(x) < 300:
            count += 1
    print("word < 300's sent: {:d}".format(count))


def data_split():
    """
    Load　data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    pos_list, neg_list = data_parse()
    positive_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in pos_list]
    negative_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in neg_list]
    # analysis
    data_analysis(positive_examples, 'pos')
    data_analysis(negative_examples, 'neg')
    # Split by words
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def data_process():
    """load train data and create the vocab processor"""
    print("Loading data...")
    x_text, y = data_split()
    # Build vocabulary
    # max_document_length = max([len(x) for x in x_text])
    max_document_length = 300
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    text_list = []
    for text in x_text:
        text_list.append(' '.join(text))
    vocab_processor.fit(text_list)
    # expand vocab by w2v
    # vocab_processor = w2v.expand_vocab(text_list, vocab_processor)
    x = np.array(list(vocab_processor.transform(text_list)))
    # Randomly shuffle data
    # np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    # return x_shuffled, y_shuffled, vocab_processor
    return x, y, vocab_processor


def load_test_data():
    """load test data and label"""
    print("Loading test data...")
    parser = MyHTMLParser()
    parser.feed(codecs.open(test_data_with_label, "r", "utf-8").read())
    test_list = data_list[1:]
    data_list.clear()
    label = np.array(label_list)
    test_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in test_list]
    data_analysis(test_examples, 'test')
    return test_examples, label


def load_test_label():
    """load test data label"""
    label_list.clear()
    parser = MyHTMLParser()
    parser.feed(codecs.open(test_data_with_label, "r", "utf-8").read())
    return np.array(label_list)


def eval(file):
    with open(file, "r", encoding="utf-8") as f:
        pre_label = np.array([int(line[1]) for line in csv.reader(f)])
    label = load_test_label()
    tp = len([i for i in range(len(label)) if label[i] == 1 and pre_label[i] == 1])
    fp = len([i for i in range(len(label)) if label[i] == 0 and pre_label[i] == 1])
    fn = len([i for i in range(len(label)) if label[i] == 1 and pre_label[i] == 0])
    tn = len([i for i in range(len(label)) if label[i] == 0 and pre_label[i] == 0])
    p = float(tp) / (tp + fp)
    r = float(tp) / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print("TP:{}, FP:{}, FN:{}, TN:{}, P:{}, R:{}, F1:{}".format(tp, fp, fn, tn, p, r, f1))


# def data_random():
#     # Randomly shuffle data
#     np.random.seed(10)
#     shuffle_indices = np.random.permutation(np.arange(len(y)))
#     x_shuffled = x[shuffle_indices]
#     y_shuffled = y[shuffle_indices]
#
#     # Split train/test set
#     # TODO: This is very crude, should use cross-validation
#     dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
#     x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#     y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#     print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#     print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# just test
if __name__ == '__main__':
    # data_process()
    load_test_label()
