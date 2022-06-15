import os
import torch
import pandas as pd
import logging

# from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
from utils import load_file_picke
from model import SSTG
from constants import *
from dataset_gen import SSTGDataset, MyCollate
from trainer import Trainer
from constants import DEVICE

logger = logging.getLogger(__name__)
device = torch.device(DEVICE)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("TRAINING WITH BATCH SIZE: {BATCH_SIZE}")
    train_df = pd.read_csv("./dataset/new_train.csv")
    val_df = pd.read_csv("./dataset/new_val.csv")

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


    train_dataset = SSTGDataset(dataset=train_df,
                                word_tokenizer=word_tokenizer,
                                char_tokenizer=char_tokenizer,
                                bos_token_id=tokenizer_config["bos_token_ids"],
                                eos_token_id=tokenizer_config["eos_token_ids"])
    
    val_dataset = SSTGDataset(dataset=val_df,
                                word_tokenizer=word_tokenizer,
                                char_tokenizer=char_tokenizer,
                                bos_token_id=tokenizer_config["bos_token_ids"],
                                eos_token_id=tokenizer_config["eos_token_ids"])


    trainer = Trainer(device=device,
                    n_words=len(word_tokenizer.word_index),
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    model_dir="./checkpoint/",
                    checkpoint=CHECKPOINT,
                    model=model,
                    resume_epoch=0,
                    epochs=20,
                    # resume=True,
                    batch_size=BATCH_SIZE)

    trainer.train()
    # trainer._eval()


if __name__ == "__main__":
    main()
