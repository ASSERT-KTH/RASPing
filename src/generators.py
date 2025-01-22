from typing import Counter
import numpy as np
from collections import Counter


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


def generate_shuffle_dyck1(acceptedTokens, maxSeqLength):
    for ii in range(
        3
    ):  # Ensures that roughly one out of eight sequences has an odd length
        inputLength = np.random.randint(
            2, maxSeqLength + 1
        )  # Uniformly distributed between 2 and max length
        if inputLength % 2 == 0:
            break

    inputSeq = []
    tokenCount = {"(": 0, ")": 0}
    tokenProb = np.zeros(
        len(acceptedTokens)
    )  # Live probabilty distribution to more evenly distribute the balanced and unblanaced sequences
    tokenProb[1] = 1 / (inputLength + 1)
    tokenProb[0] = 1 - tokenProb[1]

    # Build the sequence token by token and ensuring the probability of drawing a balanced sequence is always higher than drawing an unbalanced sequence
    for ind in range(inputLength):
        t = np.random.choice(acceptedTokens, 1, p=tokenProb)[0]
        tokenCount[t] += 1
        inputSeq.append(t)

        tokenDiff = tokenCount["("] - tokenCount[")"]
        if tokenDiff == 0:  # High probability of begining paranthesis if balanced
            tokenProb[1] = 1 / (inputLength + 1)
            tokenProb[0] = 1 - tokenProb[1]
        elif (
            tokenDiff > 0
        ):  # High probability of end paranthesis if more begining paranthesis
            tokenProb[0] = 1 / ((inputLength + 1) * tokenDiff)
            tokenProb[1] = 1 - tokenProb[0]
        else:  # High probability of begining paranthesis if more end paranthesis
            tokenProb[1] = 1 / ((inputLength + 1) * (-tokenDiff))
            tokenProb[0] = 1 - tokenProb[1]

    # Checks for balance
    balanceCounter = 0
    for t in inputSeq:
        if t == "(":
            balanceCounter += 1
        else:
            balanceCounter -= 1
        if balanceCounter < 0:
            break

    if balanceCounter != 0:
        outputSeq = [0] * len(inputSeq)
    else:
        outputSeq = [1] * len(inputSeq)

    inputSeq.insert(0, "BOS")
    outputSeq.insert(0, "BOS")

    return (inputSeq, outputSeq)


def generate_shuffle_dyck2(acceptedTokens, maxSeqLength):
    for ii in range(
        3
    ):  # Ensures that roughly one out of eight sequences has an odd length
        inputLength = np.random.randint(
            2, maxSeqLength + 1
        )  # Uniformly distributed between 2 and max length
        if inputLength % 2 == 0:
            break

    inputSeq = []
    tokenCount = {"(": 0, ")": 0, "{": 0, "}": 0}
    tokenProb = np.zeros(
        len(acceptedTokens)
    )  # Live probabilty distribution to more evenly distribute the balanced and unblanaced sequences
    tokenProb[1] = 1 / ((inputLength + 1) * 2)
    tokenProb[0] = 1 / 2 - tokenProb[1]
    tokenProb[3] = tokenProb[1]
    tokenProb[2] = tokenProb[0]

    # Build the sequence token by token and ensuring the probability of drawing a balanced sequence is always higher than drawing an unbalanced sequence
    for ind in range(inputLength):
        t = np.random.choice(acceptedTokens, 1, p=tokenProb)[0]
        tokenCount[t] += 1
        inputSeq.append(t)

        tokenDiff1 = tokenCount["("] - tokenCount[")"]
        tokenDiff2 = tokenCount["{"] - tokenCount["}"]
        if (
            tokenDiff1 == 0 and tokenDiff2 == 0
        ):  # High probability of begining paranthesis if balanced
            tokenProb[1] = 1 / ((inputLength + 1) * 2)
            tokenProb[0] = 1 / 2 - tokenProb[1]
            tokenProb[3] = tokenProb[1]
            tokenProb[2] = tokenProb[0]
        # High probability of end paranthesis if more begining paranthesis
        elif tokenDiff2 > 0 and tokenDiff1 > 0:
            tokenProb[0] = 1 / ((inputLength + 1) * tokenDiff1 * 2)
            tokenProb[2] = 1 / ((inputLength + 1) * tokenDiff2 * 2)
            tokenProb[1] = 1 / 2 - tokenProb[0]
            tokenProb[3] = 1 / 2 - tokenProb[2]
        elif tokenDiff1 > 0 and tokenDiff2 == 0:
            tokenProb[1] = 1 - 1 / ((inputLength + 1) * tokenDiff1)
            split = 1 - tokenProb[1]  # The reminder of probability to distribute
            tokenProb[2] = split - split / (
                (inputLength + 1)
            )  # More likely to start a new parenthesis than break sequence
            split = split - tokenProb[2]
            tokenProb[0] = split / 2
            tokenProb[3] = split / 2
        elif tokenDiff2 > 0 and tokenDiff1 == 0:
            tokenProb[3] = 1 - 1 / ((inputLength + 1) * tokenDiff2)
            split = 1 - tokenProb[3]  # The reminder of probability to distribute
            tokenProb[0] = split - split / (
                (inputLength + 1)
            )  # More likely to start a new parenthesis than break sequence
            split = split - tokenProb[0]
            tokenProb[1] = split / 2
            tokenProb[2] = split / 2
        # High probability of begining paranthesis if more end paranthesis
        elif tokenDiff2 < 0 and tokenDiff1 < 0:
            tokenProb[1] = 1 / ((inputLength + 1) * (-tokenDiff1) * 2)
            tokenProb[3] = 1 / ((inputLength + 1) * (-tokenDiff2) * 2)
            tokenProb[0] = 1 / 2 - tokenProb[1]
            tokenProb[2] = 1 / 2 - tokenProb[3]
        elif tokenDiff1 < 0 and tokenDiff2 == 0:
            tokenProb[0] = 1 - 1 / ((inputLength + 1) * (-tokenDiff1))
            split = 1 - tokenProb[0]  # The reminder of probability to distribute
            tokenProb[2] = split - split / (
                (inputLength + 1)
            )  # More likely to start a new parenthesis than break sequence
            split = split - tokenProb[2]
            tokenProb[1] = split / 2
            tokenProb[3] = split / 2
        elif tokenDiff2 < 0 and tokenDiff1 == 0:
            tokenProb[2] = 1 - 1 / ((inputLength + 1) * (-tokenDiff2))
            split = 1 - tokenProb[2]  # The reminder of probability to distribute
            tokenProb[0] = split - split / (
                (inputLength + 1)
            )  # More likely to start a new parenthesis than break sequence
            split = split - tokenProb[0]
            tokenProb[1] = split / 2
            tokenProb[3] = split / 2
        # Higher probability to balance the sequence if currently unbalanced
        elif tokenDiff1 > 0 and tokenDiff2 < 0:
            tokenProb[1] = 1 / ((inputLength + 1) * tokenDiff1 * 2)
            tokenProb[2] = 1 / ((inputLength + 1) * (-tokenDiff2) * 2)
            tokenProb[0] = 1 / 2 - tokenProb[1]
            tokenProb[3] = 1 / 2 - tokenProb[2]
        elif tokenDiff2 > 0 and tokenDiff1 < 0:
            tokenProb[3] = 1 / ((inputLength + 1) * tokenDiff2 * 2)
            tokenProb[0] = 1 / ((inputLength + 1) * (-tokenDiff1) * 2)
            tokenProb[1] = 1 / 2 - tokenProb[0]
            tokenProb[2] = 1 / 2 - tokenProb[3]

    # Checks for balance
    balanceCounter = [0, 0]
    for t in inputSeq:
        if t == "(":
            balanceCounter[0] += 1
        if t == ")":
            balanceCounter[0] -= 1
        if t == "{":
            balanceCounter[1] += 1
        if t == "}":
            balanceCounter[1] -= 1

        if balanceCounter[0] < 0 or balanceCounter[1] < 0:
            break

    if balanceCounter[0] != 0 or balanceCounter[1] != 0:
        outputSeq = [0] * len(inputSeq)
    else:
        outputSeq = [1] * len(inputSeq)

    inputSeq.insert(0, "BOS")
    outputSeq.insert(0, "BOS")

    return (inputSeq, outputSeq)


GENERATORS = {
    "reverse": generate_reverse,
    "hist": generate_hist,
    "sort": generate_sort,
    "most-freq": generate_most_freq,
    "shuffle_dyck1": generate_shuffle_dyck1,
    "shuffle_dyck2": generate_shuffle_dyck2,
}
