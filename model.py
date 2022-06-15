import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch import nn, Tensor
from constants import DEVICE

device = torch.device(DEVICE)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#During training, we need a subsequent word mask that will prevent model to look into the future words when making predictions.
def generate_square_mask(sequence_size: int):
    mask = (torch.triu(torch.ones((sequence_size, sequence_size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_source_mask(src: Tensor, mask_token_id: int):
    src_mask = (src == mask_token_id)
    return src_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, 
                dropout: float = 0.1, 
                max_len: int = 256):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(100000) / d_model))
        self.position_encoding = torch.zeros(max_len, d_model).to(device)
        self.position_encoding[:, 0::2] = torch.sin(position * div_term)
        self.position_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x: Tensor) -> Tensor:
        x += self.position_encoding[:x.size(1)]
        return self.dropout(x)


class CharEncoderTransformers(nn.Module):
    def __init__(self, n_chars: int, 
                mask_token_id: int, 
                d_model: int = 256, 
                d_hid: int = 256, 
                n_head: int = 4,
                n_layers: int = 4,
                dropout: float = 0.2):
        super(CharEncoderTransformers, self).__init__()
        self.position_encoding = PositionalEncoding(d_model, dropout, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.char_embedding = nn.Embedding(n_chars, d_model)
        self.d_model = d_model
        self.max_char = 200
        self.linear_char = nn.Linear(self.max_char * self.d_model, self.d_model)
        self.mask_token_id = mask_token_id
        self.init_weight()

    def init_weight(self):
        init_range = 0.1
        self.char_embedding.weight.data.uniform_(-init_range, init_range)

    def merge_embedding(self, embeddings: Tensor, sequence_split, mode='linear') -> Tensor:
        """
        :param embeddings: chars embedding [batch_size, length_seq, d_hid]
        :param sequence_split: number character for each word list[int]
        :param mode: calculate average or add embedding
        :return: [batch_size, num_words, embedding_dim]
        """
        original_sequence_split = sequence_split.copy()
        sequence_split = [value + 1 for value in sequence_split]  # plus space
        sequence_split[-1] -= 1  # remove for the last token
        embeddings = embeddings[:sum(sequence_split)]
        embeddings = torch.split(embeddings, sequence_split, dim=0)
        embeddings = [embedd[:-1, :] if i != (len(sequence_split) - 1) else embedd for i, embedd in
                      enumerate(embeddings)]
        if mode == 'avg':
            embeddings = pad_sequence(embeddings, padding_value=0, batch_first=True)  # n_word*max_length*d_hid
            seq_splits = torch.tensor(original_sequence_split).reshape(-1, 1).to(device)
            outs = torch.div(torch.sum(embeddings, dim=1), seq_splits)
        elif mode == 'add':
            embeddings = pad_sequence(embeddings, padding_value=0, batch_first=True)  # n_word*max_length*d_hid
            outs = torch.sum(embeddings, dim=1)
        elif mode == 'linear':
            embeddings =[
                torch.cat(
                    (
                        embedding_tensor.reshape(-1),
                        torch.tensor(
                            [0] * (self.max_char - embedding_tensor.size(0)) * self.d_model,
                            dtype=torch.long
                        ).to(device)
                    )
                )
                for embedding_tensor in embeddings
            ]
            embeddings = torch.stack(embeddings, dim=0)
            outs = self.linear_char(embeddings)
        else:
            raise Exception('Not Implemented')
        return outs

    def forward(self, src: Tensor,
                batch_splits,
                src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None
                ) -> Tensor:
        """
        :param src: char token ids [batch_size, max_len(setence_batch)]
        :param batch_splits:
        :param src_mask:
        :param src_key_padding_mask: mask pad token
        :return: word embedding after combine from char embedding [batch_size*n_words*d_hid]
        """
        src_embeddings = self.char_embedding(src)  # batch_size * len_seq * embedding_dim
        src_embeddings = self.position_encoding(src_embeddings)
        if src_mask is None or src_mask.size(0) != src.size(1):
            src_mask = generate_square_mask(src.size(1))

        if src_key_padding_mask is None:
            src_key_padding_mask = generate_source_mask(src, self.mask_token_id)

        outputs = self.transformer_encoder(
            src_embeddings.transpose(0, 1),
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        ).transpose(0, 1)  # batch_size*len(sentence)*d_hid
        outputs = pad_sequence(
            [self.merge_embedding(embedding, sequence_split) for embedding, sequence_split in
             zip(outputs, batch_splits)],
            padding_value=0,
            batch_first=True
        )
        return outputs


class Decoder(nn.Module):
    def __init__(self, n_words, d_hid, dropout):
        super(Decoder, self).__init__()
        self.d_hid = d_hid
        self.lstm = nn.LSTMCell(self.d_hid, self.d_hid)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(self.d_hid, n_words)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, states):
        h, c = self.lstm(x, states)
        output = self.tanh(h)
        # output = self.out(output)
        output = self.softmax(self.out(output))
        return output, (h, c)

    def initHidden(self):
        return torch.zeros(1, 1, self.d_hid, device=device)


class SSTG(nn.Module):
    def __init__(self, n_words: int, 
                n_chars: int, 
                mask_token_id: int,
                # tokenizer,
                bos_token_id,
                eos_token_id,
                pad_token_id=0,
                # device,
                # use_detection_context: bool = True, 
                d_model: int = 512, 
                d_hid: int = 768,
                n_head: int = 12, 
                n_layers: int = 12, 
                dropout: float = 0.2,
                tag_size: int = 4):
        super(SSTG, self).__init__()
        self.n_words = n_words
        self.n_chars = n_chars
        self.d_model = d_model
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # self.tokenizer = tokenizer

        self.position_encoding = PositionalEncoding(d_model, dropout, 256)
        self.char_transformer_encoder = CharEncoderTransformers(n_chars, mask_token_id)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model + self.char_transformer_encoder.d_model, n_head, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.word_embedding = nn.Embedding(n_words, d_model)

        # self.use_detection_context = use_detection_context
        self.tag_size = tag_size

        # self.tags = nn.Linear(d_hid, self.tag_size)
        self.tags = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(d_hid, self.tag_size))

        # decoder generator
        # self.resize_linear = nn.Linear(d_model, d_model + self.char_transformer_encoder.d_model)
        self.resize_linear = nn.Linear(d_model + self.char_transformer_encoder.d_model, d_model)
        self.decoder = Decoder(n_words, d_model, dropout)
        self.init_weight()


    def init_weight(self):
        init_range = 0.1
        self.word_embedding.weight.data.uniform_(-init_range, init_range)


    def forward(self, 
                src_word_error_ids: Tensor,
                src_char_ids: Tensor,
                target: Tensor,
                source_splits,
                src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None
                ):
        src_word_embeddings_ = self.word_embedding(src_word_error_ids)
        src_word_embeddings = self.position_encoding(src_word_embeddings_)
        src_words_from_chars = self.char_transformer_encoder(src_char_ids, source_splits)  # batch_size*n_words*d_model_char
        src_word_embeddings = torch.cat((src_word_embeddings, src_words_from_chars), dim=-1)  # batch_size*n_words*(d_model_char+d_model_word)

        if src_key_padding_mask is None:
            src_key_padding_mask = generate_source_mask(src_word_error_ids, self.mask_token_id)

        outputs = self.transformer_encoder(
            src_word_embeddings.transpose(0, 1),  # n_words * batch_size * hidden_size
            src_key_padding_mask=src_key_padding_mask  # batch_size * n_words * hidden_size
        ).transpose(0, 1)  # batch_size * n_words * d_hid

        # Tags 
        tag_logits = self.tags(outputs)

        batch_size, src_len, max_added_len = target.size()
        
        # Generation
        outputs = self.resize_linear(outputs)
        if target.size(-1) <= 3 and target.size(0) != 1:
            raise ValueError("Check the preprocess, must be the form of '<sep> ... </sep>' ")
        else:
            batch_h = []
            batch_c = []
            batch_word_in = []

            for bidx in range(batch_size):
                for sidx in range(src_len):
                    if target[bidx][sidx][0] not in [2, 3]:
                        continue
                    else:
                        batch_h.append(outputs[bidx][sidx].unsqueeze(0))
                        batch_c.append(outputs[bidx][sidx].unsqueeze(0))

                        batch_word_in.append(target[bidx][sidx][1:-1].unsqueeze(0))
                        # batch_word_la.append(target[bidx][sidx][2:].unsqueeze(0))

            batch_h = torch.cat(batch_h, dim=0)
            batch_c = torch.cat(batch_c, dim=0)
            batch_word_in = torch.cat(batch_word_in, dim=0)

            final_decoded = []

            # use teacher forcing
            for xidx in range(batch_word_in.size(1)):

                batch_x_word = self.word_embedding(batch_word_in[:, xidx])
                # batch_x_word = self.resize_linear(batch_x_word)

                decoder_output, _ = self.decoder(x=batch_x_word, states=(batch_h, batch_c))
                final_decoded.append(decoder_output)
            
            final_decoded = torch.stack(final_decoded, dim=1)
        return tag_logits, final_decoded


    def _init_inputs(self, 
                     src_word_error_ids: Tensor, 
                     src_char_ids: Tensor,
                     source_splits):
        state = {}

        batch_size = src_char_ids.size(0)

        src_word_embeddings_ = self.word_embedding(src_word_error_ids)
        src_word_embeddings = self.position_encoding(src_word_embeddings_)
        src_words_from_chars = self.char_transformer_encoder(src_char_ids, source_splits)
        src_word_embeddings = torch.cat((src_word_embeddings, src_words_from_chars), dim=-1)
        src_key_padding_mask = generate_source_mask(src_word_error_ids, self.mask_token_id)

        outputs = self.transformer_encoder(
            src_word_embeddings.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask
        ).transpose(0, 1)

        tag_logits = self.tags(outputs)
        tag_pred = torch.argmax(tag_logits, dim=-1)
        # print(tag_pred)
        # print(tag_pred.size())

        outputs = self.resize_linear(outputs)
        bidx_sidx_to_idx = {}
        cnt = 0
        # generator
        batch_h = []
        batch_c = []
        for bidx in range(tag_pred.size(0)):
            for sidx in range(tag_pred.size(1)):
                if tag_pred[bidx][sidx] not in [2, 3]:
                    continue
                else:
                    bidx_sidx_to_idx[(bidx, sidx)] = cnt
                    cnt += 1
                    batch_h.append(outputs[bidx][sidx].unsqueeze(0))
                    batch_c.append(outputs[bidx][sidx].unsqueeze(0))

        state["GEN"] = True
        if not batch_h:
            state["incorrect"] = False
        else:
            batch_h = torch.cat(batch_h, dim=0)
            batch_c = torch.cat(batch_c, dim=0)
            state["incorrect"] = True
            state["batch_size"] = batch_size
            state["batch_h"] = batch_h
            state["batch_c"] = batch_c
            state["bidx_sidx_to_idx"] = bidx_sidx_to_idx

        return state, tag_pred
    

    def _decode(self, state):
        batch_x = state["batch_x"] 
        batch_h = state["batch_h"]
        batch_c = state["batch_c"]
        batch_x = self.word_embedding(batch_x).squeeze(1)
        

        decoder_output, (batch_h, batch_h) = self.decoder(x=batch_x, states=(batch_h, batch_c))

        return decoder_output, state


    def _generate(self, step_fn, state, max_add_len, device, n_beams=1):
        pad_id = self.pad_token_id
        bos_id = self.bos_token_id
        unfinished_sents = torch.ones(state["batch_h"].size(0), dtype=torch.long, device=device)
        sent_lengths = unfinished_sents.new(state["batch_h"].size(0)).fill_(max_add_len)

        predictions = bos_id * torch.ones(state["batch_h"].size(0), 1, dtype=torch.long, device=device)

        for step in range(1, max_add_len + 1):
            pre_ids = predictions[:, -1:]
            state["batch_x"] = pre_ids
            decoded_output, state = step_fn(state)
            preds = torch.argmax(decoded_output, dim=-1).squeeze(-1)  # (batch_size, )
            if self.eos_token_id is not None:
                tokens_to_add = preds * unfinished_sents + pad_id * (1 - unfinished_sents)
            else:
                tokens_to_add = preds
            predictions = torch.cat([predictions, tokens_to_add.unsqueeze(dim=-1)], dim=-1)
            if self.eos_token_id is not None:
                eos_in_sents = tokens_to_add == self.eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, step)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break
        results = {
            "preds": predictions,
            "bidx_sidx_to_idx": state['bidx_sidx_to_idx']
        }
        return results


    def _generate_beam(self, step_fn, state, max_add_len, device, n_beams, length_average=True, length_penalty=0.2):
        def repeat(var, times):
            if isinstance(var, list):
                return [repeat(x, times) for x in var]
            elif isinstance(var, dict):
                return {k: repeat(v, times) for k, v in var.items()}
            elif isinstance(var, torch.Tensor):
                var = torch.unsqueeze(var, 1)
                expand_times = [1] * len(var.shape)
                expand_times[1] = times
                dtype = var.dtype
                var = var.to(torch.float).repeat(expand_times)
                shape = [var.size(0) * var.size(1)] + list(var.size())[2:]
                var = torch.reshape(var, shape).to(dtype)
                return var
            else:
                return var

        def gather(var, idx):
            if isinstance(var, list):
                return [gather(x, idx) for x in var]
            elif isinstance(var, dict):
                rlt = {}
                for k, v in var.items():
                    rlt[k] = gather(v, idx)
                return rlt
            elif isinstance(var, torch.Tensor):
                out = torch.index_select(var, dim=0, index=idx)
                return out
            else:
                return var

        pad_id = self.pad_token_id
        bos_id = self.bos_token_id
        eos_id = self.eos_token_id
        # batch_size = state["batch_size"]
        b_size = state["batch_h"].size(0)
        vocab_size = self.n_words
        beam_size = n_beams
        # print("BEAM SEARCH")
        # pos_index = torch.arange(batch_size, dtype=torch.long, device=device) * beam_size
        pos_index = torch.arange(b_size, dtype=torch.long, device=device) * beam_size
        pos_index = pos_index.unsqueeze(1)  # (batch_size, 1)

        predictions = torch.ones(b_size, beam_size, 1, dtype=torch.long, device=device) * bos_id
        # print(predictions)
        # print(predictions.size())
        # initial input
        state["batch_x"] = predictions[:, 0]

        # (batch_size, vocab_size)
        scores, state = step_fn(state)
        
        eos_penalty = np.zeros(vocab_size, dtype="float32")
        eos_penalty[eos_id] = -1e10
        eos_penalty = torch.tensor(eos_penalty, device=device)

        scores_after_end = np.full(vocab_size, -1e10, dtype="float32")
        scores_after_end[pad_id] = 0
        scores_after_end = torch.tensor(scores_after_end, device=device)

        scores = scores + eos_penalty

        # preds: (batch_size, beam_size)
        # sequence_scores: (batch_size, beam_size)
        # initialize beams
        sequence_scores, preds = torch.topk(scores, beam_size)
        # print("asdasd")
        # print(preds)
        predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)
        # print(predictions)
        state = repeat(state, beam_size)

        for step in range(2, max_add_len + 1):
            pre_ids = predictions[:, :, -1:]
            state["batch_x"] = torch.reshape(pre_ids, shape=[b_size * beam_size, 1])

            scores, state = step_fn(state)

            # Generate next
            # scores: (batch_size, beam_size, vocab_size)
            scores = torch.reshape(scores, shape=(b_size, beam_size, vocab_size))

            # previous tokens is pad or eos
            pre_eos_mask = (pre_ids == eos_id).float().to(device) + (pre_ids == pad_id).float().to(device)
            if pre_eos_mask.sum() == beam_size * b_size:
                # early stopping
                break
            scores = scores * (1 - pre_eos_mask) + pre_eos_mask.repeat(1, 1, vocab_size) * scores_after_end

            sequence_scores = sequence_scores.unsqueeze(2)

            if length_average:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 - 1 / step)
                sequence_scores = sequence_scores * scaled_value
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 / step)
                scores = scores * scaled_value
            if length_penalty > 0.0:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                               (math.pow((4 + step) / (5 + step), length_penalty))
                sequence_scores = scaled_value * sequence_scores
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                               (math.pow(1 / (5 + step), length_penalty))
                scores = scores * scaled_value

            # broadcast: every sequence combines with every potential word
            scores = scores + sequence_scores
            scores = scores.reshape(b_size, beam_size * vocab_size)

            # update beams
            topk_scores, topk_indices = torch.topk(scores, beam_size)
            parent_idx = topk_indices // vocab_size  # (batch_size, beam_size)
            preds = topk_indices % vocab_size

            # gather state / sequence_scores
            parent_idx = parent_idx + pos_index
            parent_idx = parent_idx.view(-1)
            state = gather(state, parent_idx)
            sequence_scores = topk_scores

            predictions = predictions.reshape(b_size * beam_size, step)
            predictions = gather(predictions, parent_idx)
            predictions = predictions.reshape(b_size, beam_size, step)
            predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)

        pre_ids = predictions[:, :, -1]
        pre_eos_mask = (pre_ids == eos_id).float().to(device) + (pre_ids == pad_id).float().to(device)
        sequence_scores = sequence_scores * pre_eos_mask + (1 - pre_eos_mask) * -1e10

        _, indices = torch.sort(sequence_scores, dim=1)
        indices = indices + pos_index
        indices = indices.view(-1)
        sequence_scores = torch.reshape(sequence_scores, [b_size * beam_size])
        predictions = torch.reshape(predictions, [b_size * beam_size, -1])
        sequence_scores = gather(sequence_scores, indices)
        predictions = torch.index_select(predictions, 0, indices)
        sequence_scores = torch.reshape(sequence_scores, [b_size, beam_size])
        predictions = torch.reshape(predictions, [b_size, beam_size, -1])

        results = {
            "preds": predictions[:, -1],
            "bidx_sidx_to_idx": state['bidx_sidx_to_idx']
        }
        return results


    def infer(self,
              max_add_len,
              src_word_error_ids: Tensor, 
              src_char_ids: Tensor,
              source_splits,
              n_beams=1):
        
        self.eval()
        results = {}
        device = src_word_error_ids.device
        state, tag_pred = self._init_inputs(src_word_error_ids, src_char_ids, source_splits)

        results["tag_pred"] = tag_pred
        results["word_errors"] = src_word_error_ids
        if state["incorrect"]:
            if n_beams == 1:
                gen_results = self._generate(self._decode, 
                                            state, 
                                            max_add_len, 
                                            device)
            else:
                gen_results = self._generate_beam(self._decode, 
                                                    state, 
                                                    max_add_len, 
                                                    device, 
                                                    n_beams)


            results["gen_results"] = gen_results
                
        return results
            
        # results.update(output)