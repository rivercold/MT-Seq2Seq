__author__ = 'yuhongliang324'

import random
import time
import math
import argparse

import dynet as dy
from dynet import LSTMBuilder
import numpy
from util import *


class Attention():

    # define dynet model for the encoder-decoder model
    def __init__(self, train_src_file, train_tgt_file,
                 num_layers=1, embed_size=150, hidden_size=128, attention_size=128, load_from=None, beam_search=False):
        self.num_layers = num_layers
        self.embed_size, self.hidden_size, self.attention_size = embed_size, hidden_size, attention_size

        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.src_token_to_id, self.src_id_to_token, self.src_sent_vecs, self.src_vocab_size = read_file(train_src_file)
        self.tgt_token_to_id, self.tgt_id_to_token, self.tgt_sent_vecs, self.tgt_vocab_size = read_file(train_tgt_file,
                                                                                                        target=True)
        self.src_sent_vecs, self.tgt_sent_vecs = sort_by_length(self.src_sent_vecs, self.tgt_sent_vecs)

        if beam_search:
            self.LP = compute_length_prob(self.src_sent_vecs, self.tgt_sent_vecs)

        if load_from is not None:
            [self.l2r_builder, self.r2l_builder, self.dec_builder,
             self.src_lookup, self.tgt_lookup, self.W_y, self.b_y,
             self.W_eh, self.W_hh, self.W1_att_e, self.W1_att_f, self.w2_att] = self.model.load(load_from)
            print "Finish loading"
        else:
            self.l2r_builder = LSTMBuilder(self.num_layers, self.embed_size, self.hidden_size, self.model)
            self.r2l_builder = LSTMBuilder(self.num_layers, self.embed_size, self.hidden_size, self.model)
            self.dec_builder = LSTMBuilder(self.num_layers, self.embed_size + self.hidden_size * 2, self.hidden_size,
                                           self.model)
            self.src_lookup = self.model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
            self.tgt_lookup = self.model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
            self.W_y = self.model.add_parameters((self.tgt_vocab_size, 3 * self.hidden_size))
            self.b_y = self.model.add_parameters((self.tgt_vocab_size,))
            self.W_eh = self.model.add_parameters((self.embed_size, self.hidden_size * 2))
            self.W_hh = self.model.add_parameters((self.hidden_size, self.hidden_size * 2))

            self.W1_att_e = self.model.add_parameters((self.attention_size, self.hidden_size))
            self.W1_att_f = self.model.add_parameters((self.attention_size, 2 * self.hidden_size))
            self.w2_att = self.model.add_parameters((1, self.attention_size))

        self.batch_size = 1

    def encode_single_direction(self, builder, src_sent_vec):
        enc_state = builder.initial_state()
        embeds = [dy.lookup(self.src_lookup, cID) for cID in src_sent_vec]
        outputs = enc_state.transduce(embeds)
        '''
        H_f = dy.concatenate_cols(outputs)
        encoded_h = outputs[-1]'''
        return outputs

    def encode_single_direction_batch(self, builder, pad_batch):
        enc_state = builder.initial_state()
        embeds_batch = [dy.lookup_batch(self.src_lookup, pad) for pad in pad_batch]
        outputs_batch = enc_state.transduce(embeds_batch)
        return outputs_batch

    def encode(self, src_sent_vec):
        outputs_l2r = self.encode_single_direction(self.l2r_builder, src_sent_vec)
        outputs_r2l = self.encode_single_direction(self.r2l_builder, src_sent_vec[::-1])
        outputs_r2l = outputs_r2l[::-1]
        encoded_h = dy.concatenate([outputs_l2r[-1], outputs_r2l[-1]])  # (2 * hidden_size,)
        H_f_l2r = dy.concatenate_cols(outputs_l2r)
        H_f_r2l = dy.concatenate_cols(outputs_r2l)
        H_f = dy.concatenate([H_f_l2r, H_f_r2l])  # (2 * hidden_size, num_step)

        W_eh = dy.parameter(self.W_eh)
        W_hh = dy.parameter(self.W_hh)
        encoded = W_eh * encoded_h  # (embed_size,)
        encoded_h = W_hh * encoded_h  # (hidden_size,)

        return H_f, encoded, encoded_h

    def encode_batch(self, vec_batch):
        startID, stopID = self.src_token_to_id['<S>'], self.src_token_to_id['</S>']
        pad_batch, lengths, maxLen = make_pad_bidirection(vec_batch, startID, stopID)
        outputs_l2r_batch = self.encode_single_direction_batch(self.l2r_builder, pad_batch)
        outputs_r2l_batch = self.encode_single_direction_batch(self.r2l_builder, pad_batch[::-1])
        outputs_r2l_batch = outputs_r2l_batch[::-1]

        encoded_h_batch = dy.concatenate([outputs_l2r_batch[-1], outputs_r2l_batch[-1]])  # (2 * hidden_size, batch_size)
        H_f_l2r_batch = dy.concatenate_cols(outputs_l2r_batch)  # (hidden_size, num_step, batch_size)
        H_f_r2l_batch = dy.concatenate_cols(outputs_r2l_batch)  # (hidden_size, num_step, batch_size)
        H_f_batch = dy.concatenate([H_f_l2r_batch, H_f_r2l_batch])  # (2 * hidden_size, num_step, batch_size)

        W_eh = dy.parameter(self.W_eh)
        W_hh = dy.parameter(self.W_hh)
        encoded_batch = W_eh * encoded_h_batch  # (embed_size, batch_size)
        encoded_h_batch = W_hh * encoded_h_batch  # (hidden_size, batch_size)

        return H_f_batch, encoded_batch, encoded_h_batch

    def __attention_mlp(self, H_f, h_e, W1_att_e, W1_att_f, w2_att):

        # Calculate the alignment score vector
        a_t = dy.tanh(dy.colwise_add(W1_att_f * H_f, W1_att_e * h_e))
        a_t = w2_att * a_t
        a_t = a_t[0]
        alignment = dy.softmax(a_t)
        c_t = H_f * alignment
        return c_t

    def __attention_mlp_batch(self, H_f_batch, h_e_batch, W1_att_e, W1_att_f, w2_att):
        # H_f_batch: (2 * hidden_size, num_step, batch_size)
        # h_e_batch: (hidden_size, batch_size)

        a_t_batch = dy.tanh(dy.colwise_add(W1_att_f * H_f_batch, W1_att_e * h_e_batch))  # (attention_size, num_step, batch_size)
        a_t_batch = w2_att * a_t_batch  # (1, num_step, batch_size)
        a_t_batch = a_t_batch[0]  # (num_step, batch_size)
        alignment_batch = dy.softmax(a_t_batch)  # (num_step, batch_size)
        c_t_batch = H_f_batch * alignment_batch  # (2 * hidden_size, batch_size)
        return c_t_batch

    # Training step over a single sentence pair
    def __step(self, src_sent_vec, tgt_sent_vec):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_e = dy.parameter(self.W1_att_e)
        W1_att_f = dy.parameter(self.W1_att_f)
        w2_att = dy.parameter(self.w2_att)

        losses = []

        H_f, encoded, encoded_h = self.encode(src_sent_vec)

        # Set initial decoder state to the result of the encoder
        dec_state = self.dec_builder.initial_state()
        start = True
        c_t = None
        # Calculate losses for decoding
        for (cID, nID) in zip(tgt_sent_vec, tgt_sent_vec[1:]):
            if start:
                embed = encoded
                h_e = encoded_h
                c_t = self.__attention_mlp(H_f, h_e, W1_att_e, W1_att_f, w2_att)
                start = False
            else:
                embed = dy.lookup(self.tgt_lookup, cID)
            dec_state = dec_state.add_input(dy.concatenate([embed, c_t]))
            h_e = dec_state.output()
            c_t = self.__attention_mlp(H_f, h_e, W1_att_e, W1_att_f, w2_att)
            y_star = W_y * dy.concatenate([h_e, c_t]) + b_y
            p = dy.softmax(y_star)
            loss = -dy.log(dy.pick(p, nID))
            losses.append(loss)

        loss = dy.esum(losses)

        return loss, len(losses)

    # Training step over a single sentence pair
    def __step_batch(self, src_sent_vec_batch, tgt_sent_vec_batch):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_e = dy.parameter(self.W1_att_e)
        W1_att_f = dy.parameter(self.W1_att_f)
        w2_att = dy.parameter(self.w2_att)

        losses = []

        H_f_batch, encoded_batch, encoded_h_batch = self.encode_batch(src_sent_vec_batch)

        # Set initial decoder state to the result of the encoder
        dec_state = self.dec_builder.initial_state()
        start = True
        c_t_batch = None
        # Calculate losses for decoding
        stopID = self.tgt_token_to_id['</S>']
        pad_batch, lengths, maxLen = make_pad(tgt_sent_vec_batch, stopID)
        for i in xrange(maxLen - 1):
            cID_batch, nID_batch = pad_batch[i], pad_batch[i + 1]
            if start:
                embed_batch = encoded_batch
                h_e_batch = encoded_h_batch
                c_t_batch = self.__attention_mlp_batch(H_f_batch, h_e_batch, W1_att_e, W1_att_f, w2_att)
                start = False
            else:
                embed_batch = dy.lookup_batch(self.tgt_lookup, cID_batch)
            dec_state = dec_state.add_input(dy.concatenate([embed_batch, c_t_batch]))
            h_e_batch = dec_state.output()
            c_t_batch = self.__attention_mlp(H_f_batch, h_e_batch, W1_att_e, W1_att_f, w2_att)
            y_star = W_y * dy.concatenate([h_e_batch, c_t_batch]) + b_y  # (voc_size, batch_size)
            loss = dy.pickneglogsoftmax_batch(y_star, nID_batch)
            loss = dy.reshape(loss, (self.batch_size,))  # (batch_size,)
            for j in xrange(self.batch_size):
                if lengths[j] > i + 1:
                    losses.append(dy.pick(loss, j))
        loss = dy.esum(losses) / self.batch_size

        return loss, sum(lengths)



    def translate_beam_sentence(self, src_sent_vec, max_len=50 ,beam_width=3):
        dy.renew_cg()
        def softmax(x):
            scoreMatExp = numpy.exp(numpy.asarray(x))
            return scoreMatExp / scoreMatExp.sum(0)
        def top_no_stop_list(rank_ids,stopID,beam_size):
            count, top_list = 0, []
            for i in rank_ids:
                if (i%self.tgt_vocab_size) != stopID:
                    top_list.append(i)
                    count += 1
                if count == beam_size:
                    return top_list

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_e = dy.parameter(self.W1_att_e)
        W1_att_f = dy.parameter(self.W1_att_f)
        w2_att = dy.parameter(self.w2_att)
        stopID, dec_state = self.tgt_token_to_id['</S>'], None
        H_f, encoded, encoded_h = self.encode(src_sent_vec)
        max_seq = (float('-inf'), ["<S>"])
        # dec_sate/h_e(only the first), c_t, embed, log_prob,tran_seq
        beam_list = [(encoded_h, None, 0, ["<S>"]) for _ in range(beam_width)]
        # Decoder - # Set the intial state to the result of the encoder
        iter = 0
        while iter < max_len:
            dec_state_list, c_t_list, y_star_list = [],[], []
            for i, state in enumerate(beam_list):
                [dec_state, c_t, log_prob, trans_seq] = state
                if iter == 0:
                    c_t = self.__attention_mlp(H_f, dec_state, W1_att_e, W1_att_f, w2_att)
                    dec_state = self.dec_builder.initial_state()
                    embed =  encoded
                else:
                    cID = self.tgt_token_to_id[trans_seq[-1]]
                    embed = dy.lookup(self.tgt_lookup, cID)
                dec_state = dec_state.add_input(dy.concatenate([embed, c_t]))
                h_e = dec_state.output()
                c_t = self.__attention_mlp(H_f, h_e, W1_att_e, W1_att_f, w2_att)
                dec_state_list.append(dec_state)
                c_t_list.append(c_t)
                y_star = numpy.log(softmax((W_y * dy.concatenate([h_e, c_t]) + b_y).npvalue()))
                y_star_list.append(y_star)
                # Get probability distribution for the next word to be generated
                end_prob = log_prob + y_star[stopID]
                p = y_star + log_prob
                lt = len(trans_seq)
                #print lt, end_prob/float(lt)
                if end_prob/float(lt) > max_seq[0]:
                    max_seq = (end_prob/float(lt), trans_seq)
                if i == 0:
                    beam_p = p
                else:
                    beam_p = numpy.concatenate([beam_p,p])
            temp = numpy.argpartition(- beam_p, 2 * beam_width)
            top_width_ids = top_no_stop_list(temp, stopID, beam_width)
            tmp = []
            for rank, pid in enumerate(top_width_ids):
                state_id = pid // self.tgt_vocab_size
                token_id = pid % self.tgt_vocab_size
                tmp.append((dec_state_list[state_id], c_t_list[state_id], beam_list[state_id][2] +
                            y_star_list[state_id][token_id], beam_list[state_id][3]+[self.tgt_id_to_token[token_id]]))
            beam_list = tmp
            iter += 1
        return ' '.join(max_seq[1][1:])

    def translate_sentence(self, src_sent_vec, max_len=50):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_e = dy.parameter(self.W1_att_e)
        W1_att_f = dy.parameter(self.W1_att_f)
        w2_att = dy.parameter(self.w2_att)

        H_f, encoded, encoded_h = self.encode(src_sent_vec)

        # Decoder
        trans_sentence = ['<S>']
        cID = self.tgt_token_to_id[trans_sentence[0]]

        # Set the intial state to the result of the encoder
        dec_state = self.dec_builder.initial_state()
        start = True
        c_t = None
        while len(trans_sentence) < max_len:
            if start:
                embed = encoded
                h_e = encoded_h
                c_t = self.__attention_mlp(H_f, h_e, W1_att_e, W1_att_f, w2_att)
                start = False
            else:
                embed = dy.lookup(self.tgt_lookup, cID)
            dec_state = dec_state.add_input(dy.concatenate([embed, c_t]))
            h_e = dec_state.output()
            c_t = self.__attention_mlp(H_f, h_e, W1_att_e, W1_att_f, w2_att)
            y_star = W_y * dy.concatenate([h_e, c_t]) + b_y
            # Get probability distribution for the next word to be generated
            p = dy.softmax(y_star).npvalue()
            # Find the word corresponding to the best id
            cw = self.tgt_id_to_token[numpy.argmax(p)]
            if cw == '</S>':
                break
            trans_sentence.append(cw)
            cID = self.tgt_token_to_id[cw]

        return ' '.join(trans_sentence[1:])

    def train(self, test_src_file, test_tgt_file, num_epoch=20, report_iter=100, beam_search=False, save=False, start_epoch=0):
        src_sent_vecs_test = read_test_file(test_src_file, self.src_token_to_id)
        tgt_sentences_test = read_test_file(test_tgt_file)

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
                if (j) % report_iter == 0:
                    loss_avg /= report_iter
                    print 'epoch=%d, iter=%d, loss=%f' % (i + 1, j + 1, loss_avg)
                    loss_avg = 0.
                    src_sents = [src_sent_vecs_test[k] for k in randIndex]
                    tgt_sents = [tgt_sentences_test[k] for k in randIndex]
                    self.test(src_sents, tgt_sents)
            if save:
                ctime = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
                self.save_model('Attention_epoch_{0}_layer{1}_hidden_{2}_embed_{3}_att_{4}_{5}.model'
                                .format(i + 1 + start_epoch, self.num_layers, self.hidden_size, self.embed_size,
                                        self.attention_size, ctime))

    def train_batch(self, test_src_file, test_tgt_file, num_epoch=20, batch_size=20, beam_search=False, save=False,
                    start_epoch=0):
        src_sent_vecs_test = read_test_file(test_src_file, self.src_token_to_id)
        tgt_sentences_test = read_test_file(test_tgt_file)
        self.batch_size = batch_size
        report_iter = 100 // batch_size

        num_train, num_test = len(self.src_sent_vecs), len(src_sent_vecs_test)
        num_iter = int(math.ceil(num_train / float(self.batch_size)))

        # Separate batches
        src_sent_vec_batches = []
        tgt_sent_vec_batches = []
        for i in xrange(num_iter):
            start, end = i * self.batch_size, min((i + 1) * self.batch_size, num_train)
            src_sent_vec_batches.append(self.src_sent_vecs[start: end])
            tgt_sent_vec_batches.append(self.tgt_sent_vecs[start: end])
        z = zip(src_sent_vec_batches, tgt_sent_vec_batches)
        random.shuffle(z)
        src_sent_vec_batches = [item[0] for item in z]
        tgt_sent_vec_batches = [item[1] for item in z]

        randIndex = random.sample(xrange(num_test), 5)
        loss_avg = 0.
        for i in xrange(num_epoch):
            for j in xrange(num_iter):
                lens = [len(sent) for sent in src_sent_vec_batches[j]]
                self.batch_size = len(src_sent_vec_batches[j])
                # print 'Lens:', min(lens), max(lens)
                loss, total_word = self.__step_batch(src_sent_vec_batches[j], tgt_sent_vec_batches[j])
                loss_val = loss.value() * self.batch_size / total_word
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
                ctime = time.strftime("%m-%d-%H-%M-%S",time.gmtime())
                self.save_model('Attention_batch_epoch_{0}_layer{1}_hidden_{2}_embed_{3}_att_{4}_{5}.model'
                                .format(i + 1 + start_epoch, self.num_layers, self.hidden_size, self.embed_size,
                                        self.attention_size, ctime))

    def test(self, src_sent_vecs_test, tgt_sentences_test):
        num_test = len(src_sent_vecs_test)
        for i in xrange(num_test):
            trans_sent = self.translate_sentence(src_sent_vecs_test[i])
            print trans_sent + '|\t|' + tgt_sentences_test[i]

    def save_model(self, file_name):
        folder_path = "../models"
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        theta = [self.l2r_builder, self.r2l_builder, self.dec_builder, self.src_lookup, self.tgt_lookup,
                 self.W_y, self.b_y, self.W_eh, self.W_hh, self.W1_att_e, self.W1_att_f, self.w2_att]
        self.model.save(file_path, theta)
        print 'saved to {0}'.format(file_path)

    def pred(self, test_src_file, test_tgt_file, beam=False, pred_file=None):
        src_sent_vecs_test = read_test_file(test_src_file, self.src_token_to_id)
        tgt_sentences_test = read_test_file(test_tgt_file)
        if pred_file is None:
            ctime = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
            pred_file = './results/res_{0}.txt'.format(ctime)
        writer = open(pred_file, 'w')
        num_test = len(src_sent_vecs_test)
        for i in xrange(num_test):
            if (i+1) % 10 == 0:
                print "eval num {0}".format(i+1)
            if beam:
                trans_sent = self.translate_beam_sentence(src_sent_vecs_test[i])
            else:
                trans_sent = self.translate_sentence(src_sent_vecs_test[i])
            print 'Translation:', trans_sent
            print 'Target Sent:', tgt_sentences_test[i]
            print
            writer.write(trans_sent + '\n')
        writer.close()

def save_model(model, file_path):
    folder_path = "../models"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    model_file_path = os.path.join(folder_path, file_path)
    model.save(model_file_path)
    print 'saved to {0}'.format(model_file_path)


def test1():
    att = Attention(train_de, train_en, num_layers=2)
    att.train(valid_de, valid_en, save=True)


def test2():
    att = Attention(toy_train_de, toy_train_en, load_from='../models/Attention_epoch_1_layer1_hidden_128_embed_150_att_128_02-21-00-31-15.model')
    att.train(toy_test_de, toy_test_en, num_epoch=100, save=False)


def test3():
    att = Attention(train_de, train_en, num_layers=2)
    att.train_batch(valid_de, valid_en, save=True)


def test4():
    att = Attention(train_de, train_en, num_layers=2, load_from='../models/LSTM_epoch_4_layer2_hidden_128_embed_150_att_128_02-20-16-59-09')
    att.train_batch(valid_de, valid_en, save=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', type=bool, default=False, help='Whether or not use batch training (default: False)')
    parser.add_argument('-layer', type=int, default=2, help='Number of LSTM layers (default: 2)')
    parser.add_argument('-embed', type=int, default=200, help='Embedding size (default: 200)')
    parser.add_argument('-hid', type=int, default=128, help='Hidden size (default: 128)')
    parser.add_argument('-att', type=int, default=128, help='Attention size (default: 128)')
    parser.add_argument('-load', type=str, default=None, help='Model path to load (default: None)')
    parser.add_argument('-save', type=bool, default=True, help='Whether or not save the model'
                                                               ' during training (default: True)')
    parser.add_argument('-se', type=int, default=0, help='Starting epoch, used for continue training'
                                                         ' from a certain node (default: 0)')
    parser.add_argument('-bs', type=int, default=20, help='Batch size (default: 20)')
    parser.add_argument('--dynet-mem', type=int, default=3072)
    parser.add_argument('--dynet-gpu-ids', type=int, default=3)
    parser.add_argument('-beam', default=False, help='Whether or not use Beam search in translation (default: False)')
    parser.add_argument('-beam-width', default=3, help='Beam width (default: 3)')
    parser.add_argument('-pred', type=bool, default=False, help='Only prediction without training (default: False)')
    parser.add_argument('-train_en', type=str, default=None, help='Target sentence file for training (default: None)')
    parser.add_argument('-train_de', type=str, default=None, help='Source sentence file for training (default: None)')
    parser.add_argument('-test_en', type=str, default=None, help='Target sentence file for testing (default: None)')
    parser.add_argument('-test_de', type=str, default=None, help='Source sentence file for testing (default: None)')
    parser.add_argument('-result', type=str, default=None, help='Result translation file for target testing sentences'
                                                                ' (default: None)')
    args = parser.parse_args()
    if args.train_de is None:
        args.train_de = train_de
    if args.train_en is None:
        args.train_en = train_en
    if args.test_de is None:
        args.test_de = valid_de
    if args.test_en is None:
        args.test_en = valid_en

    att = Attention(args.train_de, args.train_en, num_layers=args.layer, embed_size=args.embed,
                    hidden_size=args.hid, attention_size=args.att, load_from=args.load, beam_search=args.beam)
    if args.pred:
        print "Start translation!"
        att.pred(args.test_de, args.test_en, beam=args.beam, pred_file=args.result)
    else:
        print "Start training!"
        if args.batch:
            att.train_batch(args.test_de, args.test_en, save=args.save, start_epoch=args.se, batch_size=args.bs)
        else:
            att.train(args.test_de, args.test_en, save=args.save, start_epoch=args.se)

if __name__ == '__main__':
    main()
