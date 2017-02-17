'''
author = Keyang Xu
'''

import numpy as np
import math
from collections import defaultdict

wids =defaultdict(lambda : len(wids))

def read_corpus(file_name):
    lines = open(file_name,"r").readlines()
    for line in lines:
        words = line.strip().split()
        for word in words:
            wid = wids[word]


def train(file_name):
    total_num = 0
    lines = open(file_name,"r").readlines()
    uni_dict = defaultdict(int)
    bi_dict = defaultdict(int)
    for line in lines:
        sentence = line.strip().split()
        total_num += len(sentence)
        for i in range(len(sentence)):
            uni_dict[sentence[i]] += 1
            if i >= 1:
                bi_dict[sentence[i-1]+" "+sentence[i]] += 1

    for bigram, count in bi_dict.iteritems():
        w1 = bigram.split()[0]
        bi_dict[bigram] = float(count)/float(uni_dict[w1])

    for word, count in uni_dict.iteritems():
        uni_dict[word] = float(count)/float(total_num)

    return uni_dict, bi_dict

def test(file_name,uni_dict, bi_dict,language="E"):
    lines = open(file_name,"r").readlines()
    total_num = 0
    total_log_prob = 0.0
    for line in lines:
        sentence = line.strip().split()
        total_num += len(sentence)
        for i in range(len(sentence)):
            word = sentence[i]
            if i == 0:
                p_word = ""
            else:
                p_word = sentence[i-1]
            prob = compute_prob(p_word,word,uni_dict,bi_dict,language)
            total_log_prob += math.log(prob)

    total_log_prob /= float(total_num)
    total_log_prob *= -1.0
    perplexity = math.exp(total_log_prob)

    print "perplexity is {}".format(perplexity)

    return perplexity

def compute_prob(p_w, w, uni_dict, bi_dict, language="E"):
    if language == "E":
        alpha_unk, alpha1, alpha2 = e_alpha_unk, e_alpha_1, e_alpha_2
    elif language == "D":
        alpha_unk, alpha1, alpha2 = d_alpha_unk, d_alpha_1, d_alpha_2
    else:
        raise

    prob = alpha1 * uni_dict[w]+ alpha2 * bi_dict["{0} {1}".format(p_w,w)] + alpha_unk * unk_prob
    return prob


if __name__ == "__main__":
    language = "D"
    if language == "E":
        train_file = "../Data/train.en-de.tok.en"
        test_file = "../Data/valid.en-de.tok.en"
    else:
        train_file = "../Data/train.en-de.tok.de"
        test_file = "../Data/valid.en-de.tok.de"
    uni_dict, bi_dict = train(train_file)
    perplexity = test(test_file,uni_dict, bi_dict, language)