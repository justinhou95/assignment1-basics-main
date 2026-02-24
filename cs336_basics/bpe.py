from collections import defaultdict
import heapq
import regex as re
import multiprocessing
from tqdm import tqdm
import json
import pickle
from cs336_basics.pretokenize import pre_tokenize_parallel


def train_bpe(input_path, vocab_size, special_tokens):
    merges = []
    vocab = [bytes([i]) for i in range(256)]
    vocab += [s.encode("utf-8") for s in special_tokens]

    print("Pretokening .......")
    token_counter = pre_tokenize_parallel(input_path, special_tokens)

    print("Counting pairs .......")
    pair_counter = defaultdict(int)
    for token, count in token_counter.items():
        for i in range(len(token) - 1):
            pair = token[i : i + 2]
            pair_counter[pair] += count

    print("Training BPE .......")
    for epoch in tqdm(range(vocab_size)):
        max_pair = max(pair_counter.keys(), key=lambda pair: (pair_counter[pair], pair))
        merged_pair = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        new_counter = {}
        for token, count in token_counter.items():
            new_token = []
            i = 0
            while i < len(token):
                pair = token[i : i + 2]
                if pair == max_pair:
                    # Update the pair_counts
                    pair_counter[pair] -= count
                    if i > 0:
                        left_pair = (new_token[-1], pair[0])
                        pair_counter[left_pair] -= count
                        pair_counter[(left_pair[0], merged_pair)] += count
                    if i < len(token) - 2:
                        right_pair = token[i + 1 : i + 3]
                        pair_counter[right_pair] -= count
                        pair_counter[(merged_pair, right_pair[1])] += count
                    new_token.append(merged_pair)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_counter[tuple(new_token)] = count
        token_counter = new_counter
        vocab.append(merged_pair)
        if len(vocab) >= vocab_size:
            break

    vocab = {i: x for i, x in enumerate(vocab)}
    return vocab, merges


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    with open(f"{input_path[:-4]}_BPE_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(f"{input_path[:-4]}_BPE_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
