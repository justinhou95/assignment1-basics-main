from collections import defaultdict
from tqdm import tqdm
import pickle
from cs336_basics.pretokenize import pre_tokenize_parallel


def train_bpe(input_path, vocab_size, special_tokens):
    merges = []
    vocab = [bytes([i]) for i in range(256)]
    vocab += [s.encode("utf-8") for s in special_tokens]

    print("Pretokening .......")
    token_counter = pre_tokenize_parallel(input_path, special_tokens)

    print("Initializing pairs .......")
    merges = []
    vocab = [bytes([i]) for i in range(256)]
    vocab += [s.encode("utf-8") for s in special_tokens]
    init_vocab_len = len(vocab)

    token_pair_set = dict()
    pair_counter = defaultdict(int)
    for token, count in token_counter.items():
        s = set()
        for i in range(len(token) - 1):
            pair = token[i : i + 2]
            pair_counter[pair] += count
            s.add(pair)
        token_pair_set[token] = s

    print("Training BPE .......")
    for epoch in tqdm(range(init_vocab_len + 1, vocab_size)):
        max_pair = max(pair_counter, key=lambda p: (pair_counter[p], p))
        mp0, mp1 = max_pair
        merged_pair = mp0 + mp1
        merges.append(max_pair)

        for token, count in list(token_counter.items()):
            if max_pair not in token_pair_set[token]:
                continue  # no occurrence — token stays unchanged in dict
            modify = False
            new_token = []
            i = 0
            n = len(token)
            while i < n:
                if i + 1 < n and token[i] == mp0 and token[i + 1] == mp1:
                    modify = True
                    if new_token:  # left-neighbor update
                        left = new_token[-1]
                        lp = (left, mp0)
                        c = pair_counter[lp] - count
                        if c:
                            pair_counter[lp] = c
                        else:
                            del pair_counter[lp]
                        pair_counter[(left, merged_pair)] += count
                    if i + 2 < n:  # right-neighbor update
                        right = token[i + 2]
                        rp = (mp1, right)
                        c = pair_counter[rp] - count
                        if c:
                            pair_counter[rp] = c
                        else:
                            del pair_counter[rp]
                        pair_counter[(merged_pair, right)] += count
                    new_token.append(merged_pair)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1

            # ── in-place update (no new_counter) ──────────────────────────
            if modify:
                new_key = tuple(new_token)
                del token_counter[token]
                del token_pair_set[token]
                token_counter[new_key] = token_counter.get(new_key, 0) + count
                token_pair_set[new_key] = {
                    (new_token[j], new_token[j + 1]) for j in range(len(new_token) - 1)
                }

        del pair_counter[max_pair]  # max_pair gone from all tokens; remove stale count
        vocab.append(merged_pair)

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
