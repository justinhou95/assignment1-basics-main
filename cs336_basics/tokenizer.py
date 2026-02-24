import pickle
import token
from typing import Iterable, Iterator
import regex as re


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
        self.special_tokens = special_tokens

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

    def encode_token_bytes(self, token):
        BP_set = set(
            (token[i], token[i + 1]) for i in range(len(token) - 1)
        )  # for quick look-up
        for merge in self.merges:
            if merge in BP_set:
                new_token = []
                BP_set.clear()
                i = 0
                while i < len(token):
                    if i < len(token) - 1 and (token[i], token[i + 1]) == merge:
                        new_token.append(token[i] + token[i + 1])
                        i += 2
                    else:
                        new_token.append(token[i])
                        i += 1

                    if len(new_token) > 1:
                        BP_set.add((new_token[-2], new_token[-1]))
                token = new_token
        return [self.token_to_id[b] for b in token]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []  # encoded IDs

        delimiter = "|".join(map(re.escape, self.special_tokens))
        paragraphs = re.split(delimiter, text)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for paragraph in paragraphs:
            pre_tokens = re.finditer(PAT, paragraph)
            for pre_token in pre_tokens:
                token_bytes = tuple(
                    bytes([i]) for i in pre_token.group().encode("utf-8")
                )
                token_ids = self.encode_token_bytes(token_bytes)
                ids += token_ids
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[id] for id in ids if id in self.vocab)
        return raw_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_filepath = f"{input_path[:-4]}_BPE_vocab.pkl"
    merges_filepath = f"{input_path[:-4]}_BPE_merges.pkl"

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    ids = tokenizer.encode("i am a pig")
    print(ids)
    print(tokenizer.decode(ids))
    ids = [10123, 123123, 1232, 10000000, 213, 10, 19]
    print(tokenizer.decode(ids))
