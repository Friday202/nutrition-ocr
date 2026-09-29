# Preprocess maning take in paddle_ocr_best test and train and preprocess them
# removing non ACSII characters, removing extra spaces, etc.
# and save them in a new jsonl file

from pathlib import Path
import logging
from tqdm import tqdm

import common.helpers as helpers

if __name__ == "__main__":
    pass 