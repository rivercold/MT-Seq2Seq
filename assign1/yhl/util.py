__author__ = 'yuhongliang324'

import os
from collections import defaultdict

data_root = '../data'
train_en = os.path.join(data_root, 'train.en-de.low.filt.en')
train_de = os.path.join(data_root, 'train.en-de.low.filt.de')
valid_en = os.path.join(data_root, 'valid.en-de.low.en')
valid_de = os.path.join(data_root, 'valid.en-de.low.de')
test_en = os.path.join(data_root, 'test.en-de.low.en')
test_de = os.path.join(data_root, 'test.en-de.low.de')

toy_train_en = os.path.join(data_root, 'toy.train.en')
toy_train_de = os.path.join(data_root, 'toy.train.de')
toy_test_en = os.path.join(data_root, 'toy.test.en')
toy_test_de = os.path.join(data_root, 'toy.test.de')


def read_file(file_name, threshold=2, target=False):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    sentences = map(lambda x: x.strip().split(), lines)

    tok_count = defaultdict(int)

    for sent in sentences:
        for tok in sent:
            tok_count[tok] += 1

    tok_ID = defaultdict(int)
    if not target:  # Unknown words are set to ID = 0 for source language
        ID_tok = {0: '<UNKNOWN>'}
        curID = 1
    else:
        ID_tok = {}
        curID = 0

    for tok, cnt in tok_count.items():
        if cnt < threshold:
            continue
        tok_ID[tok] = curID
        ID_tok[curID] = tok
        curID += 1
    tok_ID['<S>'] = curID
    ID_tok[curID] = '<S>'
    curID += 1
    tok_ID['</S>'] = curID
    ID_tok[curID] = '</S>'
    curID += 1

    num_sent = len(sentences)
    sentVecs = [None for _ in xrange(num_sent)]
    for i in xrange(num_sent):
        sent = ['<S>'] + sentences[i] + ['</S>']
        num_tok = len(sent)
        vec = [0 for _ in xrange(num_tok)]
        for j in xrange(num_tok):
            vec[j] = tok_ID[sent[j]]
        sentVecs[i] = vec

    vocSize = curID
    return tok_ID, ID_tok, sentVecs, vocSize


def read_test_file(file_name, tok_ID=None):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    if tok_ID is None:
        sentences = map(lambda x: x.strip(), lines)
        return sentences
    sentences = map(lambda x: x.strip().split(), lines)

    num_sent = len(sentences)
    sentVecs = [None for _ in xrange(num_sent)]
    for i in xrange(num_sent):
        sent = ['<S>'] + sentences[i] + ['</S>']
        num_tok = len(sent)
        vec = [0 for _ in xrange(num_tok)]
        for j in xrange(num_tok):
            vec[j] = tok_ID[sent[j]]
        sentVecs[i] = vec
    return sentVecs


def test1():
    tok_ID, ID_tok, sentVecs, vocSize = read_file(train_en)
    for i in xrange(10):
        print sentVecs[i]
    print vocSize


if __name__ == '__main__':
    test1()
