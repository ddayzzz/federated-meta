import numpy as np
from torch.utils.data import Dataset
import re

# ------------------------
# 符号表

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def letter_to_index(letter):
    """
    字母转换为索引
    :param letter:
    :return:
    """
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    """
    将某个单词转换为 index
    :param word:
    :return:
    """
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# 可用于Sent40, Shakespeare


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line)  # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id] * (max_words - len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Return:
        integer list
    '''
    bag = [0] * len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


class Shakespeare(Dataset):

    def __init__(self, data, options):
        """
        这个类在读取的 pkl 为实际的数据的时候用于将 dict 格式的数据转换为 Tensor
        :param data:
        :param labels:
        """
        super(Shakespeare, self).__init__()
        sentence, label = data['x'], data['y']
        # 句子的序列, 长度为 80
        # 标记, 一个字符, 需要给出他的 index
        # 先试试能否一开始预处理, 这个和 tf 版本不一样, label 不需要 one-hot
        sentences_to_indices = [word_to_indices(word) for word in sentence]
        sentences_to_indices = np.array(sentences_to_indices)
        self.sentences_to_indices = np.array(sentences_to_indices, dtype=np.int64)
        # 处理标记, pytorch 不知道为什么要 int64 的数据作为 label 了
        self.labels = np.array([letter_to_index(letter) for letter in label], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.sentences_to_indices[index], self.labels[index]
        return data, target

