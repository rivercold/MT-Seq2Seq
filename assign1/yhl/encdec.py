__author__ = 'yuhongliang324'

import dynet as dy
from dynet import LSTMBuilder
from util import *
import numpy
import random


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

    # Training step over a single sentence pair
    def __step(self, src_sent_vec, tgt_sent_vec):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W_eh = dy.parameter(self.W_eh)

        losses = []

        # Start the rnn for the encoder
        enc_state = self.enc_builder.initial_state()

        '''
        for cID in src_sent_vec:
            embed = dy.lookup(self.src_lookup, cID)
            enc_state = enc_state.add_input(embed)
        encoded = enc_state.output()
        encoded = W_eh * encoded'''

        embeds = dy.lookup_batch(src_sent_vec, src_sent_vec)
        enc_states = enc_state.add_inputs(embeds)
        encoded = enc_states[-1].output()
        encoded = W_eh * encoded

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

    def translate_sentence(self, src_sent_vec, max_len=50):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W_eh = dy.parameter(self.W_eh)

        # Start the encoder for the sentence to be translated
        enc_state = self.enc_builder.initial_state()

        '''
        for cID in src_sent_vec:
            embed = dy.lookup(self.src_lookup, cID)
            enc_state = enc_state.add_input(embed)
        encoded = enc_state.output()
        encoded = W_eh * encoded'''
        embeds = dy.lookup_batch(self.src_lookup, src_sent_vec)
        enc_states = enc_state.add_inputs(embeds)
        encoded = enc_states[-1].output()
        encoded = W_eh * encoded

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

    def __step_batch(self, src_batch, tgt_batch):
        pass

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


def save_model(model, file_path):
    folder_path = "../models"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    model_file_path = os.path.join(folder_path, file_path)
    model.save(model_file_path)
    print 'saved to {0}'.format(model_file_path)


def test1():
    encdec = EncoderDecoder(train_de, train_en)
    encdec.train(valid_de, valid_en, save=True)


def test2():
    encdec = EncoderDecoder(toy_train_de, toy_train_en)
    encdec.train(toy_test_de, toy_test_en, num_epoch=100)

if __name__ == '__main__':
    test2()
