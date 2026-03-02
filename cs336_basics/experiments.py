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


def tokenize_main(special_tokens, input_path):
    clean_path = input_path.removesuffix(".txt")
    vocab_filepath = f"{clean_path}_BPE_vocab.pkl"
    merges_filepath = f"{clean_path}_BPE_merges.pkl"

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    chunks = []
    num_processes = multiprocessing.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        pairs = list(zip(boundaries[:-1], boundaries[1:]))
        for start, end in tqdm(pairs, total=len(pairs), desc="chunks"):
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    data = np.fromiter(
        tokenizer.encode_iterable(chunks),
        dtype=np.uint16,
    )

    print("Number of total tokens: ", len(data))
    print(f"{clean_path}_tokens.bin")
    data.tofile(f"{clean_path}_tokens.bin")

    return 0


# special_tokens = ["<|endoftext|>"]
# input_path = "./data/TinyStoriesV2-GPT4-train.txt"
# vocab_size = 10000
# tokenize_main(special_tokens, input_path)

# special_tokens = ["<|endoftext|>"]
# input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
# vocab_size = 10000
# tokenize_main(special_tokens, input_path)

if __name__ == "__main__":
    special_tokens = []
    input_path = "./data/owt_valid.txt"
    vocab_size = 32000
    train_bpe_main(special_tokens, input_path, vocab_size)
    # tokenize_main(special_tokens, input_path)

    # special_tokens = []
    # input_path = "./data/owt_train.txt"
    # vocab_size = 32000
    # tokenize_main(special_tokens, input_path)

    # train_bpe_main(special_tokens, input_path, vocab_size)
    # tokenize_main(special_tokens, input_path)
