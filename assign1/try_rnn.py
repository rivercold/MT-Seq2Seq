__author__ = 'yuhongliang324'

import random
from collections import defaultdict
from itertools import count
import sys
from dynet import *

LAYERS = 2
INPUT_DIM = 50
HIDDEN_DIM = 50

characters = list("abcdefghijklmnopqrstuvwxyz ")
characters.append("<EOS>")

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

model = Model()


srnn = SimpleRNNBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
lstm = LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

params = {}
params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
params["R"] = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
params["bias"] = model.add_parameters((VOCAB_SIZE))

# return compute loss of RNN for one sentence
def do_one_sentence(rnn, sentence):
    # setup the sentence
    renew_cg()
    s0 = rnn.initial_state()


    R = parameter(params["R"])
    bias = parameter(params["bias"])
    lookup = params["lookup"]
    sentence = ["<EOS>"] + list(sentence) + ["<EOS>"]
    sentence = [char2int[c] for c in sentence]
    s = s0
    loss = []
    for char,next_char in zip(sentence,sentence[1:]):
        s = s.add_input(lookup[char])
        probs = softmax(R*s.output() + bias)
        loss.append( -log(pick(probs,next_char)) )
    loss = esum(loss)
    return loss


# generate from model:
def generate(rnn):
    def sample(probs):
        rnd = random.random()
        for i,p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    # setup the sentence
    renew_cg()
    s0 = rnn.initial_state()

    R = parameter(params["R"])
    bias = parameter(params["bias"])
    lookup = params["lookup"]

    s = s0.add_input(lookup[char2int["<EOS>"]])
    out=[]
    while True:
        probs = softmax(R*s.output() + bias)
        probs = probs.vec_value()
        next_char = sample(probs)
        out.append(int2char[next_char])
        if out[-1] == "<EOS>": break
        s = s.add_input(lookup[next_char])
    return "".join(out[:-1]) # strip the <EOS>


# train, and generate every 5 samples
def train(rnn, sentence):
    trainer = SimpleSGDTrainer(model)
    for i in xrange(200):
        loss = do_one_sentence(rnn, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 5 == 0:
            print loss_value,
            print generate(rnn)

sentence = "a quick brown fox jumped over the lazy dog"
train(srnn, sentence)
train(srnn, "these pretzels are making me thirsty")
