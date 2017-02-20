__author__ = 'yuhongliang324'

import dynet


def load(fn):
    model = dynet.Model()
    ret = model.load(fn)
    print ret
    return model.load(fn), model


load('../models/LSTM_epoch_4_layer2_hidden_128_embed_150_att_128_02-20-16-59-09')
