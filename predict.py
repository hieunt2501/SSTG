# import os
from __future__ import annotations
import torch
# import pandas as pd
import numpy as np
import ujson
import logging
import re

from sklearn.metrics import precision_score, recall_score, fbeta_score, classification_report
# from transformers import AutoTokenizer
from utils import load_file_picke
from model import SSTG
from constants import CHAR_TOKENIZER, TAGS
from trainer import Trainer
from constants import DEVICE, WORD_TOKENIZER, CHAR_TOKENIZER, TOKENIZER_CONFIG


# logger = logging.getLogger(__name__)
device = torch.device("cpu")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_test_set(file):
    with open(file, "r", encoding="utf8") as f:
        dataset = ujson.load(f)
    return dataset


def main():
    logging.basicConfig(filename='./test_result/viwiki/test_0.log', level=logging.DEBUG, force=True)

    char_tokenizer = load_file_picke(CHAR_TOKENIZER)
    word_tokenizer = load_file_picke(WORD_TOKENIZER)
    tokenizer_config = load_file_picke(TOKENIZER_CONFIG)


    model = SSTG(
        n_words=len(word_tokenizer.word_index),
        n_chars=len(char_tokenizer.word_index),
        bos_token_id=tokenizer_config["bos_token_ids"],
        eos_token_id=tokenizer_config["eos_token_ids"],
        mask_token_id=0
    )

    test_set = read_test_set("./dataset/viwiki_split_no_punct_ignore.json")
    # result_file = "./result/test_5k_{}_beam.json"

    trainer = Trainer(device=device,
                    n_words=len(word_tokenizer.word_index),
                    model_dir="./checkpoint/",
                    checkpoint="./checkpoint/model_new/model8.pth",
                    model=model,
                    bos_token_id=tokenizer_config["bos_token_ids"],
                    eos_token_id=tokenizer_config["eos_token_ids"],
                    batch_size=1,
                    tokenizer=word_tokenizer,
                    char_tokenizer=char_tokenizer)

    # text = "Với chiến thắng của ông mở ra gần 200 năm đô hộ của người La Mã ờ vùng România"
    n_beams = [3]
    add_len = 4

    for beam in n_beams:
        predictions = []
        logging.info(f"{'-'*50}")
        logging.info(f"Beam search {beam}")
        y_pred = []
        y_true = []

        true_detection = 0
        error_detected = 0
        actual_errors = 0

        exact_correction = 0
        wrong_correction = 0
        wrong_detection = 0
        for idx, item in enumerate(test_set):
            try:
                text = item["text"]
                correct = item["correct_text"]
                annotations = item["annotations"]
                tags, preds, edits = trainer.infer(text, add_len, beam)
                predictions.append({
                    "corrected": preds[0],
                    "idx": idx,
                    "correct": correct,
                    "incorrect": text
                })
                tags = np.array(tags[0])
                tags_error = np.where(tags==1, 1, 0)

                assert len(tags)==len(annotations)
                true_tag = np.array([1 if item["is_correct"] else 0 for item in annotations])
                is_ignore = [item["ignore"] for item in annotations]
                error_index = np.where(tags_error==0)[0]

                actual_errors += len(np.where(true_tag==0)[0])

                for error_idx in error_index:
                    if true_tag[error_idx] == 1 and is_ignore[error_idx]:
                        tags_error[error_idx] = true_tag[error_idx]

                # unique, counts = np.unique(true_tag==tags_error, return_counts=True)
                # difference_dict = dict(zip(unique, counts))

                for i, j in zip(true_tag, tags_error):
                    if j==0:
                        error_detected += 1
                    if i==0 and j==0:
                        true_detection += 1

                # true_detection += difference_dict[True]
                # if False in difference_dict:
                #     error_detected += difference_dict[False]

                tok_id = 0
                logging.info(f"{'-'*50}")
                logging.info(f"Index: {idx}")
                logging.info(f"Incorrect text: {text}")
                logging.info(f"Correct text: {item['correct_text']}")
                logging.info(f"Predictions: {preds}")
                logging.info(f"{'-'*50}")
                # print(edits)
                # error_index = np.where(tags_error==0)[0]
                for error_idx in error_index:
                    if true_tag[error_idx] == 1 and not is_ignore[error_idx]:
                        wrong_detection += 1

                        if tags[error_idx] != TAGS["DELETE"]:
                            tok_id += 1
                    elif true_tag[error_idx] == 1 and is_ignore[error_idx]:
                        continue
                    else:                        
                        alternative_syllabels = annotations[error_idx]["alternative_syllables"] 

                        if tags[error_idx] != TAGS["DELETE"] and edits[tok_id] in alternative_syllabels:
                            exact_correction += 1
                            tok_id += 1
                        elif tags[error_idx] == TAGS["DELETE"] and not alternative_syllabels:
                            exact_correction += 1
                        elif tags[error_idx] == TAGS["DELETE"] and alternative_syllabels:
                            wrong_correction += 1
                        else:
                            wrong_correction += 1
                            tok_id += 1
                y_pred.append(tags_error.tolist())
                y_true.append(true_tag.tolist())
                # if idx == 100: break
                # break
            except Exception as e:
                logging.info(f"{'*'*50}")
                logging.info(f"Check index: {idx}")
                logging.warning(e)
                logging.info(f"{'*'*50}")

        logging.info("TAGS RESULTS")
        logging.info("-"*20)
        y_pred = [_ for y in y_pred for _ in y]
        y_true = [_ for y in y_true for _ in y]
        logging.info("Sklearn based score")
        logging.info(f"Classification report:\n{classification_report(y_true, y_pred)}")
        logging.info(f"Precision: {precision_score(y_true, y_pred, average='macro')}")
        logging.info(f"Recall: {recall_score(y_true, y_pred, average='macro')}")
        logging.info(f"F1: {fbeta_score(y_true, y_pred, beta=1, average='macro')}")
        logging.info(f"F0.5 {fbeta_score(y_true, y_pred, beta=0.5, average='macro')}")
        logging.info("-"*20)
        
        logging.info("VSEC based score")
        dp = true_detection / error_detected if error_detected else 1
        dr = true_detection/actual_errors if actual_errors else 1
        fd = 2 * dp * dr / (dr + dp)
        logging.info(f"Precision: {dp}")
        logging.info(f"Recall: {dr}")
        logging.info(f"F1: {fd}")
        logging.info("*"*20)

        logging.info("CORRECTION RESULT")
        logging.info("-"*20)
        logging.info("VIWKI based score")
        if exact_correction or wrong_correction:
            acc_detected = exact_correction / (exact_correction + wrong_correction)
        else:
            acc_detected = 0
        
        if exact_correction or wrong_correction or wrong_detection:
            acc_total = exact_correction / (exact_correction + wrong_correction + wrong_detection)
        else:
            acc_total = 0
        logging.info(f"Exact correction: {exact_correction}")
        logging.info(f"Wrong correction: {wrong_correction}")
        logging.info(f"Wrong detection: {wrong_detection}")
        logging.info(f"True detection: {true_detection}")
        logging.info(f"Error detected: {error_detected}")
        logging.info(f"Actual error: {actual_errors}")
        logging.info(f"Acc in detected: {acc_detected}")
        logging.info(f"Acc total: {acc_total}")

        logging.info("-"*20)
        logging.info("VSEC based score")
        cp = exact_correction / error_detected if error_detected else 1
        cr = exact_correction / actual_errors if actual_errors else 1
        fc = 2 * cr * cp / (cr + cp)
        logging.info(f"Precision: {cp}")
        logging.info(f"Recall: {cr}")
        logging.info(f"F1: {fc}")
        # print(predictions)
        # with open(result_file.format(beam), "w", encoding="utf8") as f:
        #     ujson.dump(predictions, f, ensure_ascii=False)
if __name__ == "__main__":
    main()
