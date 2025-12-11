from transformers import AutoTokenizer
from .config import TOKENIZER_ID


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

