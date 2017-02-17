from collections import defaultdict
import dynet as dy
import numpy as np
import random
import sys

class EncoderDecoder:

    # define dynet model for the encoder-decoder model
    def __init__(self, training_src, training_tgt, ...):
        self.model = dynet.Model()
        self.trainer = dy.SimpleSGDTrainer(model)

        self.src_token_to_id, self.src_id_to_token = XXXX
        self.tgt_token_to_id, self.tgt_id_to_token = XXXX

        self.enc_builder = builder(self.layers, self.embed_size, self.hidden_size, model) 
        self.dec_builder = builder(self.layers, self.embed_size, self.hidden_size, model)
        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size))
        self.b_y = model.add_parameters((self.tgt_vocab_size))

    # Training step over a single sentence pair
    def __step(self, instance):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        src_sent, tgt_sent = instance
        losses = []
        total_words = 0

        # Start the rnn for the encoder 
        enc_state = self.enc_builder.initial_state()
        for cw in src_sent:
            enc_state = XXXX 
        encoded = enc_state.output()

        # Set initial decoder state to the result of the encoder
        dec_state = self.dec_builder.initial_state([encoded])

        # Calculate losses for decoding
        for (cw, nw) in zip(tgt_sent, tgt_sent[1:]):
            dec_state = XXXX
            y_star = XXXX([b_y, W_y, dec_state.output()])
            loss = XXXX(y_star, nw)
            losses.append(loss)
            total_words += 1
 
        return dy.esum(losses), total_words

    def translate_sentence(self, sent, max_len=50):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        # Start the encoder for the sentence to be translated
        enc_state = self.enc_builder.initial_state()
        for cw in sent:
            enc_state = XXXX
        encoded = enc_state.output()

        # Decoder
        trans_sentence = ['<S>']
        cw = trans_sentence[0]

        # Set the intial state to the result of the encoder
        dec_state = self.dec_builder.initial_state([encoded])
        while len(trans_sentence) < max_len:
            dec_state = XXXX 
            y_star = XXXX([b_y, W_y, dec_state.output()])
            # Get probability distribution for the next word to be generated
            p = XXXX(y_star)
            # Find the word corresponding to the best id
            cw = self.tgt_id_to_token[XXXX]
            if cw == '</S>':
                break
            trans_sentence.append(cw)

        return ' '.join(trans_sentence[1:])

    def __step_batch(self, batch):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        src_batch = [x[0] for x in batch]
        tgt_batch = [x[1] for x in batch]

        # Encoder
        # src_batch  = [ [a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5], [c1,c2,c3,c4], ...]
        # transpose the batch into 
        #   src_cws: [[a1,b1,c1,..], [a2,b2,c2,..], ... [a5,b5,END,...]]
        #   src_len: [5,5,4,...]

        src_cws = []
        src_len = [len(sent) for sent in src_batch]

        encodings = []
        enc_state = self.enc_builder.initial_state()
        for i, cws in enumerate(src_cws):
            enc_state = XXXX # lookup_batch
            encodings.append(enc_state.output())

        # We want to extract the correct encodings for the correct timestep for each sentence,
        # then reconstruct the state so that they are the same dimensions as the decoder's state
        #   src_encodings: [[e(a)1,e(a)2,...,e(a)d], [e(2)1,e(a)2,], [e(c)], ...]
        src_encodings = []
        for i, l in enumerate(src_len):
            src_encodings.append(encodings[l-1].npvalue()[:, 0, i])

        encoded = XXX(src_encodings)
 
        losses = []
        total_words = 0
        
        # Decoder
        # tgt_batch  = [ [a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5], [c1,c2,c3,c4] ..]
        # transpose the batch into 
        #   tgt_cws: [[a1,b1,c1,..], [a2,b2,c2,..], .. [a5,b5,END, ...]]
        #   masks: [1,1,1,..], [1,1,1,..], ...[1,1,0,..]]
        tgt_cws = []
        masks = []
        total_words = XXXX

        dec_state = self.dec_builder.initial_state([encoded])
        for i, (cws, nws, mask) in enumerate(zip(tgt_cws, tgt_cws[1:], masks)):
            dec_state = XXXX #lookup_batch
            y_star = XXXX([b_y, W_y, dec_state.output()])
            loss = XXXX(y_star, nws) #pickneglogsoftmax_batch
            mask_loss = XXX(mask, loss)
            losses.append(mask_loss)

        return dy.sum_batches(dy.esum(losses)), total_words

def main():
    training_src = read_file(sys.argv[1])
    training_tgt = read_file(sys.argv[2])
    dev_src = read_file(sys.argv[3])
    dev_tgt = read_file(sys.argv[4])
    test_src = read_file(sys.argv[5])
    encdec = EncoderDecoder(model, training_src, training_tgt)

if __name__ == '__main__': main()
