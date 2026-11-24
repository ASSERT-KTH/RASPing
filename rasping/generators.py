"""Ground-truth input/output generators for the base programs.

Each generator returns `(input_seq, output_seq)` with a leading "BOS" token on both.
Random generators draw from NumPy's *global* RNG (seed it before generating; F24).
Exhaustive generators enumerate all inputs of length 2..max_len in length order.
"""

import itertools
from collections import Counter

import numpy as np


def generate_reverse(acceptedTokens, maxSeqLength):
    inputLength = np.random.randint(2, maxSeqLength + 1)
    inputSeq = list(np.random.choice(acceptedTokens, inputLength))
    outputSeq = inputSeq[::-1]

    inputSeq.insert(0, "BOS")
    outputSeq.insert(0, "BOS")

    return (inputSeq, outputSeq)


def generate_hist(acceptedTokens, maxSeqLength):
    inputLength = np.random.randint(2, maxSeqLength + 1)
    inputSeq = np.random.choice(acceptedTokens, inputLength).tolist()
    tokenCounts = Counter(inputSeq)
    outputSeq = [tokenCounts[t] for t in inputSeq]

    inputSeq.insert(0, "BOS")
    outputSeq.insert(0, "BOS")

    return (inputSeq, outputSeq)


def generate_sort(acceptedTokens, maxSeqLength):
    input_length = np.random.randint(2, maxSeqLength + 1)
    input_seq = list(np.random.choice(acceptedTokens, input_length))
    output_seq = sorted(input_seq)

    return (["BOS"] + input_seq, ["BOS"] + output_seq)


def generate_most_freq(acceptedTokens, maxSeqLength):
    inputLength = np.random.randint(2, maxSeqLength + 1)
    inputSeq = list(np.random.choice(acceptedTokens, inputLength))
    tokenCounts = Counter(inputSeq)
    outputSeq = sorted(inputSeq, key=lambda x: -tokenCounts[x])

    inputSeq.insert(0, "BOS")
    outputSeq.insert(0, "BOS")

    return (inputSeq, outputSeq)


def generate_reverse_exhaustive(acceptedTokens, maxSeqLength):
    all_pairs = []
    for length in range(2, maxSeqLength + 1):
        # Generate all possible combinations for current length
        for seq in itertools.product(acceptedTokens, repeat=length):
            inputSeq = list(seq)
            outputSeq = inputSeq[::-1]

            # Add BOS token
            inputSeq.insert(0, "BOS")
            outputSeq.insert(0, "BOS")

            all_pairs.append((inputSeq, outputSeq))

    return all_pairs


def generate_hist_exhaustive(acceptedTokens, maxSeqLength):
    all_pairs = []
    for length in range(2, maxSeqLength + 1):
        for seq in itertools.product(acceptedTokens, repeat=length):
            inputSeq = list(seq)
            tokenCounts = Counter(inputSeq)
            outputSeq = [tokenCounts[t] for t in inputSeq]

            inputSeq.insert(0, "BOS")
            outputSeq.insert(0, "BOS")

            all_pairs.append((inputSeq, outputSeq))

    return all_pairs


def generate_sort_exhaustive(acceptedTokens, maxSeqLength):
    all_pairs = []
    for length in range(2, maxSeqLength + 1):
        for seq in itertools.product(acceptedTokens, repeat=length):
            inputSeq = list(seq)
            outputSeq = sorted(inputSeq)

            inputSeq.insert(0, "BOS")
            outputSeq.insert(0, "BOS")

            all_pairs.append((inputSeq, outputSeq))

    return all_pairs


def generate_most_freq_exhaustive(acceptedTokens, maxSeqLength):
    all_pairs = []
    for length in range(2, maxSeqLength + 1):
        for seq in itertools.product(acceptedTokens, repeat=length):
            inputSeq = list(seq)
            tokenCounts = Counter(inputSeq)
            # ties keep first-occurrence order, like Tracr's make_sort_freq (F25)
            outputSeq = sorted(inputSeq, key=lambda x: -tokenCounts[x])

            inputSeq.insert(0, "BOS")
            outputSeq.insert(0, "BOS")

            all_pairs.append((inputSeq, outputSeq))

    return all_pairs


#: Keyed by the data-file program name (`most-freq`, ...).
GENERATORS = {
    "reverse": generate_reverse,
    "reverse_exhaustive": generate_reverse_exhaustive,
    "hist": generate_hist,
    "hist_exhaustive": generate_hist_exhaustive,
    "sort": generate_sort,
    "sort_exhaustive": generate_sort_exhaustive,
    "most-freq": generate_most_freq,
    "most-freq_exhaustive": generate_most_freq_exhaustive,
}


def get_generator(program: str, exhaustive: bool = False):
    """Generator for a program given by any of its names."""
    from rasping.programs import get

    key = get(program).data_name + ("_exhaustive" if exhaustive else "")
    return GENERATORS[key]
