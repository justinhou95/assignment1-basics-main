from collections import defaultdict
import multiprocessing
from tqdm import tqdm
import pickle
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.pretokenize import pre_tokenize_parallel
from cs336_basics.bpe import train_bpe
import numpy as np
from cs336_basics.train import train

from cs336_basics.tokenizer import Tokenizer


def train_bpe_main(special_tokens, input_path, vocab_size):
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    clean_path = input_path.removesuffix(".txt")
    with open(f"{clean_path}_BPE_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(f"{clean_path}_BPE_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


def tokenize_main(special_tokens, input_path, file_path):
    clean_input_path = input_path.removesuffix(".txt")
    vocab_filepath = f"{clean_input_path}_BPE_vocab.pkl"
    merges_filepath = f"{clean_input_path}_BPE_merges.pkl"
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    clean_file_path = file_path.removesuffix(".txt")
    chunks = []
    num_processes = 10000
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    pairs = list(zip(boundaries[:-1], boundaries[1:]))

    with open(file_path, "rb") as f:
        for start, end in tqdm(pairs, total=len(pairs), desc="chunks"):
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    data = np.fromiter(
        tqdm(tokenizer.encode_iterable(chunks), desc="tokenizing", unit="tok"),
        dtype=np.uint16,
    )

    print("Number of total tokens: ", len(data))
    print(f"{clean_file_path}_tokens.bin")
    data.tofile(f"{clean_file_path}_tokens.bin")

    return 0


special_tokens = ["<|endoftext|>"]
input_path = "./data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
file_path = "./data/TinyStoriesV2-GPT4-valid.txt"
# file_path = "./data/TinyStoriesV2-GPT4-train.txt"

# special_tokens = ["<|endoftext|>"]
# input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
# vocab_size = 10000

# special_tokens = ["<|endoftext|>"]
# input_path = "./data/owt_valid.txt"
# vocab_size = 32000

if __name__ == "__main__":
    # train_bpe_main(special_tokens, input_path, vocab_size)
    tokenize_main(special_tokens, input_path, file_path)
