import dynet as dy
from dynet import LSTMBuilder
from util import *
import numpy
import random
import nltk


class EncoderDecoder:

    # define dynet model for the encoder-decoder model
    def __init__(self, train_src_file, train_tgt_file, num_layers=1, embed_size=200, hidden_size=128):

        self.num_layers = num_layers
        self.embed_size, self.hidden_size = embed_size, hidden_size

        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.src_token_to_id, self.src_id_to_token, self.src_sent_vecs, self.src_vocab_size = read_file(train_src_file)
        self.tgt_token_to_id, self.tgt_id_to_token, self.tgt_sent_vecs, self.tgt_vocab_size = read_file(train_tgt_file)

        self.enc_builder = LSTMBuilder(self.num_layers, self.embed_size, self.hidden_size, self.model)
        self.dec_builder = LSTMBuilder(self.num_layers, self.embed_size+self.hidden_size, self.hidden_size, self.model)
        self.src_lookup = self.model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
        self.tgt_lookup = self.model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.W_y = self.model.add_parameters((self.tgt_vocab_size, 2*self.hidden_size))
        self.b_y = self.model.add_parameters((self.tgt_vocab_size,))
        self.W_eh = self.model.add_parameters((self.embed_size, self.hidden_size))
        self.W1_att = self.model.add_parameters((self.hidden_size, 2*self.hidden_size))
        self.W2_att = self.model.add_parameters((self.hidden_size))


    def __attention_mlp(self, h_fs_matrix, h_e):
        W1_att = dy.parameter(self.W1_att)
        W2_att = dy.parameter(self.W2_att)
        # Calculate the alignment score vector
        # Hint: Can we make this more efficient?
        a_t = compute_att_scores(W1_att, W2_att, h_fs_matrix, h_e)
        alignment = dy.softmax(a_t)
        c_t = h_fs_matrix * alignment
        return c_t

    def compute_att_scores(self,W1,W2,h_mat,h_e):
        h_concat_mat = dy.concatenate(h_mat,h_e)
        a_t = W2.T * dy.tanh(W1 * h_concat_mat)
        return a_t

    def encode(self, src_sent_vec):
        W_eh = dy.parameter(self.W_eh)
        enc_state = self.enc_builder.initial_state()
        h_fs = []
        for cID in src_sent_vec:
            embed = dy.lookup(self.src_lookup, cID)
            enc_state = enc_state.add_input(embed)
            h_fs.append(enc_state.output())
        encoded_h = enc_state.output()
        encoded_e = W_eh * encoded_h
        h_fs_matrix = transfer_to_matrix(h_fs)
        return encoded_h, encoded_e, h_fs_matrix

    # Training step over a single sentence pair
    def __step(self, src_sent_vec, tgt_sent_vec):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        losses = []

        # Start the rnn for the encoder
        encoded_h, encoded_e, h_fs = self.encode(src_sent_vec)
        # Set initial decoder state to the result of the encoder
        dec_state = self.dec_builder.initial_state()
        start = True
        # Calculate losses for decoding
        for (cID, nID) in zip(tgt_sent_vec, tgt_sent_vec[1:]):
            if start:
                h_e = encoded_h
                c_t = self.__attention_mlp(h_fs, h_e)
                x_t = dy.concatenate(c_t,encoded_e)
                start = False
            else:
                h_e = dec_state.output()
                c_t = self.__attention_mlp(h_fs,h_e)
                embed = dy.lookup(self.tgt_lookup, cID)
                x_t = dy.concatenate(c_t,embed)

            dec_state = dec_state.add_input(x_t)
            c_t_plus_1 = self.__attention_mlp(h_fs,dec_state.output())
            y_star = W_y * dy.concatenate(dec_state.output(),c_t_plus_1) + b_y
            p = dy.softmax(y_star)
            loss = -dy.log(dy.pick(p, nID))
            losses.append(loss)

        loss = dy.esum(losses)
        return loss, len(losses)

    def translate_sentence(self, src_sent_vec, max_len=50):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        # Start the encoder for the sentence to be translated
        enc_state = self.enc_builder.initial_state()

        for cID in src_sent_vec:
            embed = dy.lookup(self.src_lookup, cID)
            enc_state = enc_state.add_input(embed)
        encoded = enc_state.output()

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

    def train(self, test_src_file, test_tgt_file, num_epoch=10, report_iter=100):
        src_sent_vecs_test = read_test_file(test_src_file, self.src_token_to_id)
        tgt_sentences_test = read_test_file(test_tgt_file)

        # TODO: shuffle training set

        num_train, num_test = len(self.src_sent_vecs), len(src_sent_vecs_test)
        '''
        print len(self.src_sent_vecs), len(self.tgt_sent_vecs)
        train_data = zip(self.src_sent_vecs, self.tgt_sent_vecs)
        train_data.sort(key=lambda t: - len(t[0]))
        '''
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
            save_model(self.model,"LSTM_layer1_SGD_{0}".format(i+1))
            # TODO: save model

    def test(self, src_sent_vecs_test, tgt_sentences_test):
        num_test = len(src_sent_vecs_test)
        BLEUscores = []
        for i in xrange(num_test):
            trans_sent = self.translate_sentence(src_sent_vecs_test[i])
            print trans_sent + '|\t|' + tgt_sentences_test[i]
            score = nltk.translate.bleu_score.bleu([tgt_sentences_test[i]],trans_sent,weights=(0.25, 0.25,0.25,0.25))
            BLEUscores.append(score)
        print "BLEU score is ", numpy.mean(BLEUscores)


def save_model(model,file_path):
    folder_path = "../models"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    model_file_path = os.path.join(folder_path,file_path)
    model.save(model_file_path)
    print 'saved to {0}'.format(model_file_path)

def test1():
    encdec = EncoderDecoder(train_de, train_en)
    encdec.train(valid_de, valid_en)


def test2():
    encdec = EncoderDecoder(toy_train_de, toy_train_en)
    encdec.train(toy_test_de, toy_test_en, num_epoch=100)

if __name__ == '__main__':
    test2()
