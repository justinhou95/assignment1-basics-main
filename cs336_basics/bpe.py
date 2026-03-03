from ast import List
from collections import defaultdict
from tqdm import tqdm
import pickle
import heapq
from cs336_basics.pretokenize import pre_tokenize_parallel


class _RevKey:
    """Reverses comparison so that in a min-heap, larger keys win ties."""

    __slots__ = ("key",)

    def __init__(self, key):
        self.key = key

    def __lt__(self, other):
        return self.key > other.key

    def __le__(self, other):
        return self.key >= other.key

    def __eq__(self, other):
        return self.key == other.key

    def __gt__(self, other):
        return self.key < other.key

    def __ge__(self, other):
        return self.key <= other.key


class MaxHeapDict(dict):
    """
    A dict subclass with default value 0 and O(log n) amortized max-by-value lookup
    via a lazy-deletion heap.

    Heap entries are (-value, _RevKey(key)). Stale entries are skipped lazily in popmax.
    For equal values, the *largest* key wins (matching Python's max() tiebreaker).
    Keys must be totally ordered.

    Extra API beyond dict:
        d.popmax()   →  (key, value)   O(log n) amortized, removes the entry
    """

    def __init__(self, *args, **kwargs):
        self._heap: list = []
        super().__init__(*args, **kwargs)

    def __missing__(self, _):
        return 0

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        heapq.heappush(self._heap, (-value, _RevKey(key)))

    def __delitem__(self, key):
        super().__delitem__(key)  # heap entry becomes stale, cleaned up lazily

    def popmax(self):
        """Remove and return (key, value) with maximum value. O(log n) amortized."""
        heap = self._heap
        while heap:
            neg_val, rev_key = heapq.heappop(heap)
            key = rev_key.key
            if key in self and self[key] == -neg_val:
                super().__delitem__(key)
                return key, -neg_val
        raise KeyError("MaxHeapDict is empty")


def train_bpe(input_path, vocab_size, special_tokens):
    print("Pretokening .......")
    word_freq = pre_tokenize_parallel(input_path, special_tokens)

    print("Initializing pairs .......")
    merges = []
    vocab = [bytes([i]) for i in range(256)]
    vocab += [s.encode("utf-8") for s in special_tokens]
    n_merges = vocab_size - len(vocab)

    pair_freq = MaxHeapDict()
    pair_to_words = defaultdict(set)
    word_to_key = dict()
    for word, count in word_freq.items():
        word_to_key[word] = word
        for i in range(len(word) - 1):
            pair = word[i : i + 2]
            pair_to_words[pair].add(word)
            pair_freq[pair] += count

    print("Training BPE .......")
    for _ in tqdm(range(n_merges)):
        best_pair, _ = pair_freq.popmax()
        merged_bytes = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab.append(merged_bytes)

        for word in pair_to_words[best_pair]:
            word_count = word_freq[word]
            key = word_to_key[word]
            new_key = []
            i = 0
            key_len = len(key)
            while i < key_len:
                if (
                    i + 1 < key_len
                    and key[i] == best_pair[0]
                    and key[i + 1] == best_pair[1]
                ):
                    if new_key:  # left-neighbor update
                        pair_freq[(new_key[-1], best_pair[0])] -= word_count
                        if pair_freq[(new_key[-1], best_pair[0])] == 0:
                            del pair_freq[(new_key[-1], best_pair[0])]
                            pair_to_words[(new_key[-1], best_pair[0])].remove(word)
                        pair_freq[(new_key[-1], merged_bytes)] += word_count
                        pair_to_words[(new_key[-1], merged_bytes)].add(word)

                    if i + 2 < key_len:  # right-neighbor update
                        pair_freq[(best_pair[1], key[i + 2])] -= word_count
                        if pair_freq[(best_pair[1], key[i + 2])] == 0:
                            del pair_freq[(best_pair[1], key[i + 2])]
                            pair_to_words[(best_pair[1], key[i + 2])].remove(word)

                        pair_freq[(merged_bytes, key[i + 2])] += word_count
                        pair_to_words[(merged_bytes, key[i + 2])].add(word)
                    new_key.append(merged_bytes)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            word_to_key[word] = tuple(new_key)

        del pair_to_words[best_pair]

    vocab = {i: x for i, x in enumerate(vocab)}
    return vocab, merges


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    with open(f"{input_path[:-4]}_BPE_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(f"{input_path[:-4]}_BPE_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


# def old_train_bpe(input_path, vocab_size, special_tokens):
#     merges = []
#     vocab = [bytes([i]) for i in range(256)]
#     vocab += [s.encode("utf-8") for s in special_tokens]

#     print("Pretokening .......")
#     token_counter = pre_tokenize_parallel(input_path, special_tokens)

#     print("Counting pairs .......")
#     pair_counter = defaultdict(int)
#     for token, count in token_counter.items():
#         for i in range(len(token) - 1):
#             pair = token[i : i + 2]
#             pair_counter[pair] += count

#     print("Training BPE .......")
#     for epoch in tqdm(range(vocab_size)):
#         max_pair = max(pair_counter.keys(), key=lambda pair: (pair_counter[pair], pair))
#         merged_pair = max_pair[0] + max_pair[1]
#         merges.append(max_pair)
#         new_counter = {}
#         for token, count in token_counter.items():
#             new_token = []
#             i = 0
#             while i < len(token):
#                 pair = token[i : i + 2]
#                 if pair == max_pair:
#                     # Update the pair_counts
#                     pair_counter[pair] -= count
#                     if i > 0:
#                         left_pair = (new_token[-1], pair[0])
#                         pair_counter[left_pair] -= count
#                         pair_counter[(left_pair[0], merged_pair)] += count
#                     if i < len(token) - 2:
#                         right_pair = token[i + 1 : i + 3]
#                         pair_counter[right_pair] -= count
#                         pair_counter[(merged_pair, right_pair[1])] += count
#                     new_token.append(merged_pair)
#                     i += 2
#                 else:
#                     new_token.append(token[i])
#                     i += 1
#             new_counter[tuple(new_token)] = count
#         token_counter = new_counter
#         vocab.append(merged_pair)
#         if len(vocab) >= vocab_size:
#             break

#     vocab = {i: x for i, x in enumerate(vocab)}
#     return vocab, merges
