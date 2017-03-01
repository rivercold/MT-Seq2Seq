__author__ = 'yuhongliang324'

import random
import math

import dynet as dy
from dynet import LSTMBuilder
import numpy

from util import *


class EncoderDecoder:

    # define dynet model for the encoder-decoder model
    def __init__(self, train_src_file, train_tgt_file, num_layers=1, embed_size=200, hidden_size=128):

        self.num_layers = num_layers
        self.embed_size, self.hidden_size = embed_size, hidden_size

        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.src_token_to_id, self.src_id_to_token, self.src_sent_vecs, self.src_vocab_size = read_file(train_src_file)
        self.tgt_token_to_id, self.tgt_id_to_token, self.tgt_sent_vecs, self.tgt_vocab_size = read_file(train_tgt_file,
                                                                                                        target=True)

        self.enc_builder = LSTMBuilder(self.num_layers, self.embed_size, self.hidden_size, self.model)
        self.dec_builder = LSTMBuilder(self.num_layers, self.embed_size, self.hidden_size, self.model)
        self.src_lookup = self.model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
        self.tgt_lookup = self.model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.W_y = self.model.add_parameters((self.tgt_vocab_size, self.hidden_size))
        self.b_y = self.model.add_parameters((self.tgt_vocab_size,))
        self.W_eh = self.model.add_parameters((self.embed_size, self.hidden_size))
        self.batch_size = 1

    def encode(self, src_sent_vec):
        W_eh = dy.parameter(self.W_eh)
        enc_state = self.enc_builder.initial_state()
        embeds = [dy.lookup(self.src_lookup, cID) for cID in src_sent_vec]
        outputs = enc_state.transduce(embeds)
        encoded = outputs[-1]
        encoded = W_eh * encoded
        return encoded

    def make_mask(self, vecs):
        num_vec = len(vecs)
        stopID = self.tgt_token_to_id['</S>']
        lengths = [len(vec) for vec in vecs]
        maxLen = max(lengths)
        masks = []
        for j in xrange(maxLen):
            masks.append([(vecs[i][j] if lengths[i] > j else stopID) for i in xrange(num_vec)])
        return masks, lengths, maxLen

    def encode_batch(self, src_sent_vecs):
        W_eh = dy.parameter(self.W_eh)
        enc_state = self.enc_builder.initial_state()

        masks, lengths, maxLen = self.make_mask(src_sent_vecs)
        embeds_batch = [dy.lookup_batch(self.src_lookup, mask) for mask in masks]
        outputs_batch = enc_state.transduce(embeds_batch)
        encoded_batch = outputs_batch[-1]
        # (hidden_size, batch_size)
        encoded_batch = W_eh * encoded_batch  # (embed_size, batch_size)
        return encoded_batch

    # Training step over a single sentence pair
    def __step(self, src_sent_vec, tgt_sent_vec):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        losses = []

        encoded = self.encode(src_sent_vec)

        # Set initial decoder state to the result of the encoder
        dec_state = self.dec_builder.initial_state()
        start = True
        # Calculate losses for decoding
        for (cID, nID) in zip(tgt_sent_vec, tgt_sent_vec[1:]):
            if start:
                embed = encoded
                start = False
            else:
                embed = dy.lookup(self.tgt_lookup, cID)
            dec_state = dec_state.add_input(embed)
            y_star = W_y * dec_state.output() + b_y
            p = dy.softmax(y_star)
            loss = -dy.log(dy.pick(p, nID))
            losses.append(loss)

        loss = dy.esum(losses)

        return loss, len(losses)

    def __step_batch(self, src_sent_vecs, tgt_sent_vecs):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        losses = []
        encoded_batch = self.encode_batch(src_sent_vecs)

        dec_state = self.dec_builder.initial_state()
        start = True

        masks, lengths, maxLen = self.make_mask(tgt_sent_vecs)
        for i in xrange(maxLen - 1):
            cID_batch, nID_batch = masks[i], masks[i + 1]
            if start:
                embed_batch = encoded_batch
                start = False
            else:
                embed_batch = dy.lookup_batch(self.tgt_lookup, cID_batch)
            dec_state = dec_state.add_input(embed_batch)  # (hidden_size, batch_size)
            y_star = W_y * dec_state.output() + b_y  # (voc_size, batch_size)
            loss = dy.pickneglogsoftmax_batch(y_star, nID_batch)  # (batch_size,)
            loss = dy.reshape(loss, (self.batch_size,))
            for j in xrange(self.batch_size):
                if lengths[j] > i + 1:
                    losses.append(dy.pick(loss, j))

        loss = dy.esum(losses)
        return loss, sum(lengths)

    def translate_sentence(self, src_sent_vec, max_len=50):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        encoded = self.encode(src_sent_vec)

        # Decoder
        trans_sentence = ['<S>']
        cID = self.tgt_token_to_id[trans_sentence[0]]

        # Set the intial state to the result of the encoder
        dec_state = self.dec_builder.initial_state()
        # dec_state = dec_state.add_input(encoded)
        start = True
        while len(trans_sentence) < max_len:
            if start:
                embed = encoded
                start = False
            else:
                embed = dy.lookup(self.tgt_lookup, cID)
            dec_state = dec_state.add_input(embed)
            y_star = W_y * dec_state.output() + b_y
            # Get probability distribution for the next word to be generated
            p = dy.softmax(y_star).npvalue()
            # Find the word corresponding to the best id
            cw = self.tgt_id_to_token[numpy.argmax(p)]
            if cw == '</S>':
                break
            trans_sentence.append(cw)
            cID = self.tgt_token_to_id[cw]

        return ' '.join(trans_sentence[1:])

    def train(self, test_src_file, test_tgt_file, num_epoch=20, report_iter=100, save=False):
        src_sent_vecs_test = read_test_file(test_src_file, self.src_token_to_id)
        tgt_sentences_test = read_test_file(test_tgt_file)

        # TODO: shuffle training set

        num_train, num_test = len(self.src_sent_vecs), len(src_sent_vecs_test)
        randIndex = random.sample(xrange(num_test), 10)
        loss_avg = 0.
        for i in xrange(num_epoch):
            for j in xrange(num_train):
                loss, total_word = self.__step(self.src_sent_vecs[j], self.tgt_sent_vecs[j])
                loss_val = loss.value() / total_word
                loss_avg += loss_val
                loss.backward()
                self.trainer.update()
                if (j + 1) % report_iter == 0:
                    loss_avg /= report_iter
                    print 'epoch=%d, iter=%d, loss=%f' % (i + 1, j + 1, loss_avg)
                    loss_avg = 0.
                    src_sents = [src_sent_vecs_test[k] for k in randIndex]
                    tgt_sents = [tgt_sentences_test[k] for k in randIndex]
                    self.test(src_sents, tgt_sents)
            if save:
                save_model(self.model, 'LSTM_layer1_SGD_{0}'.format(i + 1))

    def test(self, src_sent_vecs_test, tgt_sentences_test):
        num_test = len(src_sent_vecs_test)
        for i in xrange(num_test):
            trans_sent = self.translate_sentence(src_sent_vecs_test[i])
            print trans_sent + '|\t|' + tgt_sentences_test[i]
        print

    def train_batch(self, test_src_file, test_tgt_file, num_epoch=20, batch_size=50, report_iter=2, save=False):
        self.batch_size = batch_size
        src_sent_vecs_test = read_test_file(test_src_file, self.src_token_to_id)
        tgt_sentences_test = read_test_file(test_tgt_file)

        # TODO: shuffle training set

        num_train, num_test = len(self.src_sent_vecs), len(src_sent_vecs_test)
        num_iter = int(math.ceil(num_train / float(self.batch_size)))
        randIndex = random.sample(xrange(num_test), 10)
        loss_avg = 0.
        for i in xrange(num_epoch):
            for j in xrange(num_iter):
                start, end = j * self.batch_size, min((j + 1) * self.batch_size, num_train)
                loss, total_word = self.__step_batch(self.src_sent_vecs[start: end], self.tgt_sent_vecs[start: end])
                loss_val = loss.value() / total_word
                loss_avg += loss_val
                loss.backward()
                self.trainer.update()
                if (j + 1) % report_iter == 0:
                    loss_avg /= report_iter
                    print 'epoch=%d, iter=%d/%d, loss=%f' % (i + 1, j + 1, num_iter, loss_avg)
                    loss_avg = 0.
                    src_sents = [src_sent_vecs_test[k] for k in randIndex]
                    tgt_sents = [tgt_sentences_test[k] for k in randIndex]
                    self.test(src_sents, tgt_sents)
            if save:
                save_model(self.model, 'LSTM_layer1_SGD_{0}'.format(i + 1))


def save_model(model, file_path):
    folder_path = "../models"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    model_file_path = os.path.join(folder_path, file_path)
    model.save(model_file_path, )
    print 'saved to {0}'.format(model_file_path)


def test1():
    encdec = EncoderDecoder(train_de, train_en)
    encdec.train(valid_de, valid_en, save=True)


def test2():
    encdec = EncoderDecoder(toy_train_de, toy_train_en)
    encdec.train(toy_test_de, toy_test_en, num_epoch=100)

if __name__ == '__main__':
    test1()
