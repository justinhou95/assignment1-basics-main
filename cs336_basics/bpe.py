from collections import defaultdict
import regex as re
import multiprocessing
from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(chunk):
    return re.findall(PAT, chunk)


def train_bpe(input_path, vocab_size, special_tokens):
    merge = []
    vocab = [bytes([i]) for i in range(256)]
    vocab.append(special_tokens)

    print("Pretokening .......")

    token_counter = defaultdict(int)
    with open(input_path, "rb") as f:
        num_processes = 64
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pattern = "|".join(re.escape(tok) for tok in special_tokens)
            split_chunk = re.split(pattern, chunk)
            # split_chunk is a list of paragraph or list of story
            with multiprocessing.Pool() as pool:
                pretokenized_split_chunk = pool.map(pretokenize, split_chunk)
            for tokens in pretokenized_split_chunk:
                for token in tokens:
                    utf8_encoded = token.encode("utf-8")
                    key = tuple(
                        bytes([x]) for x in utf8_encoded
                    )  # key is a tuple of bytes
                    token_counter[key] += 1

    print("Counting pairs .......")

    pair_counter = defaultdict(int)
    for token, count in token_counter.items():
        for i in range(len(token) - 1):
            pair = token[i : i + 2]
            pair_counter[pair] += count

    print("Training BPE .......")

    epoch = 0
    while len(vocab) < vocab_size:
        epoch += 1
        if epoch % 100 == 0:
            print(epoch)

        max_pair = max(pair_counter.keys(), key=lambda pair: (pair_counter[pair], pair))
        merged_pair = max_pair[0] + max_pair[1]
        merge.append(max_pair)
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

    return merge, vocab
