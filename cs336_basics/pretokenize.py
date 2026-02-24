from collections import Counter, defaultdict
import multiprocessing
import os
import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries


def pre_tokenize_chunk(args):
    # read certain chunk of the text
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pre_token_freq: defaultdict[tuple[bytes, ...], int] = defaultdict(int)

    # split the chunk to paragraphs by special tokens
    delimiter = "|".join(map(re.escape, special_tokens))
    paragraphs = re.split(delimiter, chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for paragraph in paragraphs:
        # split tokens in each paragraph, each token is a word
        pre_tokens = re.finditer(PAT, paragraph)
        for pre_token in pre_tokens:
            pre_token_bytes = tuple(
                bytes([i]) for i in pre_token.group().encode("utf-8")
            )
            pre_token_freq[pre_token_bytes] += 1
    return pre_token_freq


def pre_tokenize_parallel(file_path: str | os.PathLike, special_tokens: list[str]):
    num_processes = multiprocessing.cpu_count()
    pre_token_freq: Counter[tuple[bytes, ...]] = Counter()

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    args = [
        (file_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with multiprocessing.Pool(num_processes) as pool:
        # tqdm now sees progress
        for freq in tqdm(
            pool.imap_unordered(pre_tokenize_chunk, args),
            total=len(args),
        ):
            pre_token_freq.update(freq)

    return pre_token_freq


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    pre_tokenize_parallel(input_path, special_tokens)
