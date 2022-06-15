from data_helper.align_sentence import align_sentence
import torch
from underthesea import ner
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# from data_helper.phobert_tokenize import convert_tags
from data_helper.word_align import process_ner
from constants import TAGS


import re 

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    # text = re.search('[^\\x00-\\x7F\\x80-\\xFF\\u0100-\\u017F\\u0180-\\u024F\\u1E00-\\u1EFF]', text)
    return text


class SSTGDataset(Dataset):
    def __init__(self,
                 #  dataset_path,
                 dataset,
                 word_tokenizer,
                 char_tokenizer,
                 bos_token_id,
                 eos_token_id,
                 pad_token_id=0,
                 max_char_len=600,
                 max_src_len=150,
                 max_add_len=3):
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer
        self.dataset = dataset
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_char_len = max_char_len
        self.max_src_len = max_src_len
        self.max_add_len = max_add_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        incorrect = str(data.error_sentence).strip()
        correct = str(data.correct_sentence).strip()
        return self._prepare_data(incorrect, correct)

    def _prepare_data(self, source, target):
        data = {}
        # print(source)
        # print(target)
        source = preprocess(source)
        target = preprocess(target)
        # print(source)
        # print(target)

        data["incorrect_sentence"] = source
        data["correct_sentence"] = target

        try:
            tags, src, trg = align_sentence(source, target)
        except Exception as e:
            print(source)
            print(target)
            raise e
 
        original_char_error = " ".join(src)
        data["incorrect_sentence"] = source
        data["correct_sentence"] = original_char_error     
        error_char = self.char_tokenizer.texts_to_sequences([original_char_error])[0][:self.max_char_len]   
        error_word = self.word_tokenizer.texts_to_sequences([original_char_error])[0][:self.max_src_len]

        tags = tags[:self.max_src_len]
        
        target = [[] for _ in range(len(src))]

        for idx in range(len(src)):
            if tags[idx] in ["CHANGE", "APPEND"]:
                trg_token = trg[idx]
                add = [self.bos_token_id] + \
                        self.word_tokenizer.texts_to_sequences([trg_token])[0][:self.max_add_len-1] + \
                        [self.eos_token_id]

                if tags[idx].startswith("CHANGE|"):
                    target[idx].extend([TAGS["CHANGE"]] + add)
                else:
                    target[idx].extend([TAGS["APPEND"]] + add)
            else:
                target[idx].append(TAGS[tags[idx]])

        source_split = self._get_size_word_in_sentence(original_char_error)

        # tags, src = convert_tags(source, target)
        # tags, src = make_labels(tags, src)

        # assert src

        # original_src = " ".join(src)
        # error_word = self.word_tokenizer.texts_to_sequences([original_src])[
        #     0][:self.max_src_len]
        # error_char = self.char_tokenizer.texts_to_sequences([original_src])[
        #     0][:self.max_char_len]
        # source_split = self._get_size_word_in_sentence(original_src)

        # tags = tags[:self.max_src_len]
        # target = [[] for _ in range(len(src))]

        # for idx in range(len(src)):
        #     if tags[idx].startswith("CHANGE|") or tags[idx].startswith("APPEND|"):
        #         add = tags[idx][len("CHANGE|"):].split("<|>")
        #         add = [self.bos_token_id] + \
        #             self.word_tokenizer.texts_to_sequences([add])[0][:self.max_add_len-1] + \
        #             [self.eos_token_id]
        #         if tags[idx].startswith("CHANGE|"):
        #             target[idx].extend([TAGS["CHANGE"]] + add)
        #         else:
        #             target[idx].extend([TAGS["APPEND"]] + add)
        #     else:
        #         target[idx].append(TAGS[tags[idx]])

        data["word_errors"] = error_word
        data["char_errors"] = error_char
        data["source_split"] = source_split
        data["targets"] = target

        return data

    def _get_size_word_in_sentence(self, sentence: str):
        return [len(word) for word in sentence.split()]


class MyCollate:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        data = {}
        char_errors = []
        word_errors = [torch.tensor(
            data['word_errors'], dtype=torch.long) for data in batch]
        char_errors = [torch.tensor(
            data['char_errors'], dtype=torch.long) for data in batch]
        correct_sentences = [data["correct_sentence"] for data in batch]
        incorrect_sentences = [data["incorrect_sentence"] for data in batch]
        source_splits = [data["source_split"] for data in batch]
        word_max_len = -1
        add_word_max_len = -1

        for x in batch:
            word_max_len = max(len(x["targets"]), word_max_len)
            for t_w in x["targets"]:
                add_word_max_len = max(add_word_max_len, len(t_w))

        target = []
        for example in batch:
            _tgt_w = []
            for t_w in example["targets"]:
                _tgt_w.append(t_w + [0] * (add_word_max_len - len(t_w)))

            example_len = len(example["targets"])
            added_len = word_max_len - example_len
            target.append(_tgt_w + [[0] * add_word_max_len] * added_len)

        data["targets"] = torch.tensor(target, dtype=torch.long)
        data['char_errors'] = pad_sequence(
            char_errors, padding_value=self.pad_id, batch_first=True)
        data['word_errors'] = pad_sequence(
            word_errors, padding_value=self.pad_id, batch_first=True)
        data["correct_sentences"] = correct_sentences
        data["incorrect_sentences"] = incorrect_sentences
        data["source_splits"] = source_splits

        return data


class InferDataset(Dataset):
    def __init__(self,
                 source,
                 word_tokenizer,
                 char_tokenizer,
                 bos_token_id,
                 eos_token_id,
                 pad_token_id=0,
                 max_char_len=512,
                 max_src_len=512,
                 max_add_len=3):
        if isinstance(source, str):
            source = [source]
        self.source = source
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_char_len = max_char_len
        self.max_src_len = max_src_len
        self.max_add_len = max_add_len

    def __len__(self):
        return len(self.source)

    def _get_size_word_in_sentence(self, sentence: str):
        return [len(word) for word in sentence.split()]

    def __getitem__(self, idx):
        item = self.source[idx]
        return self._prepare_input(item)

    def _prepare_input(self, source):
        data = {}
        error_word = self.word_tokenizer.texts_to_sequences([source])[
            0][:self.max_src_len]
        error_char = self.char_tokenizer.texts_to_sequences([source])[
            0][:self.max_char_len]
        source_split = self._get_size_word_in_sentence(source)
        ner_tag = ner(source)
        _, ner_tag = process_ner(source, ner_tag)

        data["word_errors"] = error_word
        data["char_errors"] = error_char
        data["source_split"] = source_split
        data["source_sentence"] = text_to_word_sequence(source)
        data["untok_sentence"] = source
        data["ner_tag"] = ner_tag
        return data


class InferCollate:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        data = {}
        word_errors = [torch.tensor(
            data['word_errors'], dtype=torch.long) for data in batch]
        char_errors = [torch.tensor(
            data['char_errors'], dtype=torch.long) for data in batch]
        source_splits = [data["source_split"] for data in batch]
        source_sentences = [data["source_sentence"] for data in batch]
        untok_sentences = [data["untok_sentence"] for data in batch]
        ner_tags = [data["ner_tag"] for data in batch]

        data['char_errors'] = pad_sequence(
            char_errors, padding_value=self.pad_id, batch_first=True)
        data['word_errors'] = pad_sequence(
            word_errors, padding_value=self.pad_id, batch_first=True)
        data["source_splits"] = source_splits
        data["source_sentences"] = source_sentences
        data["untok_sentences"] = untok_sentences
        data["ner_tags"] = ner_tags

        return data
