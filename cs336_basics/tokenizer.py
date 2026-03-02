import pickle
import token
from typing import Iterable, Iterator
import numpy as np
import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {token: id for id, token in vocab.items()}
        self.special_tokens = special_tokens or []
        # Precompute merge priority for O(1) rank lookup during encoding.
        # Lower rank = higher priority (applied first).
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }
        self._encode_cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)

    def encode_token_bytes(self, token: bytes):
        if token in self._encode_cache:
            return self._encode_cache[token]
        token_array = [bytes([x]) for x in token]
        while len(token_array) >= 2:
            best_rank = None
            best_idx = -1
            for i in range(len(token_array) - 1):
                rank = self.merge_rank.get((token_array[i], token_array[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            token_array[best_idx : best_idx + 2] = [
                token_array[best_idx] + token_array[best_idx + 1]
            ]
        result = [self.token_to_id[b] for b in token_array]
        self._encode_cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []  # encoded IDs
        if self.special_tokens:
            delimiter = (
                "("
                + "|".join(
                    map(re.escape, sorted(self.special_tokens, key=lambda x: -len(x)))
                )
                + ")"
            )
            paragraphs = re.split(delimiter, text)
        else:
            paragraphs = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for paragraph in paragraphs:
            if paragraph in self.special_tokens:
                ids += [self.token_to_id[paragraph.encode("utf-8")]]
            else:
                pre_tokens = re.finditer(PAT, paragraph)
                for pre_token in pre_tokens:
                    token_ids = self.encode_token_bytes(
                        pre_token.group().encode("utf-8")
                    )
                    ids += token_ids
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[id] for id in ids if id in self.vocab)
        return raw_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":

    special_tokens = []
    input_path = "./data/owt_train.txt"
    vocab_filepath = f"{input_path[:-4]}_BPE_vocab.pkl"
    merges_filepath = f"{input_path[:-4]}_BPE_merges.pkl"

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    file_path = "./data/owt_valid.txt"

    def read_chunks(file_path, chunk_size=1024 * 1024):  # 1 MB at a time
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk.decode("utf-8", errors="ignore")

    chunks = read_chunks(file_path)

    data = np.fromiter(
        tokenizer.encode_iterable(chunks),
        dtype=np.uint16,
    )

    print(len(data))
    data.tofile("./data/owt_valid_tokens.bin")

    # special_tokens = ["<|endoftext|>"]
    # input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    # vocab_filepath = f"{input_path[:-4]}_BPE_vocab.pkl"
    # merges_filepath = f"{input_path[:-4]}_BPE_merges.pkl"

    # tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # chunks = []
    # file_path = "./data/TinyStoriesV2-GPT4-train.txt"
    # num_processes = 1000
    # with open(file_path, "rb") as f:
    #     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    #     pairs = list(zip(boundaries[:-1], boundaries[1:]))
    #     for start, end in tqdm(pairs, total=len(pairs), desc="chunks"):
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         chunks.append(chunk)

    # data = np.fromiter(
    #     tokenizer.encode_iterable(chunks),
    #     dtype=np.uint16,
    # )

    # print(len(data))
    # data.tofile("./data/TinyStoriesV2-GPT4-train_tokens.bin")

    # chunks = []
    # file_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    # num_processes = 1000
    # with open(file_path, "rb") as f:
    #     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    #     pairs = list(zip(boundaries[:-1], boundaries[1:]))
    #     for start, end in tqdm(pairs, total=len(pairs), desc="chunks"):
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         chunks.append(chunk)

    # data = np.fromiter(
    #     tokenizer.encode_iterable(chunks),
    #     dtype=np.uint16,
    # )

    # print(len(data))
    # data.tofile("./data/TinyStoriesV2-GPT4-valid_tokens.bin")
