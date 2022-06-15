TAGS = {
    "DELETE": 0,
    "KEEP": 1,
    "CHANGE": 2,
    "APPEND": 3
}

CHAR_TOKENIZER = './pretrained/char_tokenizer.pkl'
WORD_TOKENIZER = './pretrained/word_tokenizer.pkl'
TOKENIZER_CONFIG = "./pretrained/tokenizer_info.pkl"

PHOBERT_TOKENIZER = "./pretrained/phobert"

DEVICE = "cuda"
CHECKPOINT = ""
BATCH_SIZE = 32