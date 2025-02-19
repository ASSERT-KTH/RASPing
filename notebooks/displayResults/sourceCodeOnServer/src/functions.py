import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.rasp import rasp

from .model import Model


# Return all the accepted model names and their corresponding accepted inputs
def getAcceptedNamesAndInput():
    return {
        "reverse": [
            "a",
            "b",
            "c",
            "d",
            "e",
        ],  # Tokens doesn't matter much. Only the quantity influnce the results due to encoding (I think)
        "hist": [
            "a",
            "b",
            "c",
            "d",
        ],  # Tokens doesn't matter much. Only the quantity influnce the results due to encoding (I think)
        "sort": [
            1,
            2,
            3,
            4,
            5,
            6,
        ],  # [0,1,2,3,4,5,6]    Seems to fail sometimes if 0 is included (irrespktive of if 0 is in the failed input or not, don't know why)
        "most-freq": [1, 2, 3, 4, 5],
        "shuffle_dyck1": ["(", ")"],
        "shuffle_dyck2": ["(", ")", "{", "}"],
    }


# Generate a data set based on "name" with "size" samples and a max sequence length of "maxSeqLength"
def generateData(name: str, maxSeqLength: int, size: int, removeDuplicates=False):
    data = [None] * size

    acceptedNamesAndInput = getAcceptedNamesAndInput()

    match name:
        case "reverse":
            acceptedTokens = acceptedNamesAndInput[name]

            for i in range(size):
                inputLength = np.random.randint(
                    2, maxSeqLength + 1
                )  # Uniformly distributed between 2 and max length

                inputSeq = []
                outputSeq = []
                for t in np.random.choice(acceptedTokens, inputLength):
                    inputSeq.append(t)
                    outputSeq.insert(0, t)
                inputSeq.insert(0, "BOS")
                outputSeq.insert(0, "BOS")

                data[i] = (inputSeq, outputSeq)

        case "hist":
            acceptedTokens = acceptedNamesAndInput[name]

            for i in range(size):
                inputLength = np.random.randint(
                    2, maxSeqLength + 1
                )  # Uniformly distributed between 2 and max length

                inputSeq = []
                tokenCounter = dict(
                    zip(acceptedTokens, [0] * len(acceptedTokens))
                )  # Counter built during generating input
                for t in np.random.choice(acceptedTokens, inputLength):
                    inputSeq.append(t)
                    tokenCounter[t] += 1

                outputSeq = []
                for t in inputSeq:  # Fill output according to token counter
                    outputSeq.append(tokenCounter[t])

                inputSeq.insert(0, "BOS")
                outputSeq.insert(0, "BOS")

                data[i] = (inputSeq, outputSeq)

        case "sort":
            acceptedTokens = acceptedNamesAndInput[name]

            for i in range(size):
                inputLength = np.random.randint(
                    2, maxSeqLength + 1
                )  # Uniformly distributed between 2 and max length

                inputSeq = []
                outputSeq = []
                for t in np.random.choice(acceptedTokens, inputLength):
                    inputSeq.append(t)
                    outputSeq.append(t)

                inputSeq.insert(0, "BOS")
                outputSeq.sort()
                outputSeq.insert(0, "BOS")

                data[i] = (inputSeq, outputSeq)

        case (
            "most-freq"
        ):  # sort based on most frequent token with original position as tie breaker
            acceptedTokens = acceptedNamesAndInput[name]

            for i in range(size):
                inputLength = np.random.randint(
                    2, maxSeqLength + 1
                )  # Uniformly distributed between 2 and max length

                inputSeq = []
                tempSeq = []
                tokenCounter = dict(
                    zip(acceptedTokens, [0] * len(acceptedTokens))
                )  # Counter built during generating input
                for t in np.random.choice(acceptedTokens, inputLength):
                    inputSeq.append(t)
                    tokenCounter[t] += 1
                    tempSeq.append(t)

                tempSeq.sort(
                    key=(lambda x: -tokenCounter[x])
                )  # Sort the list in descending order of frequency

                outputSeq = tempSeq

                # Groups the tokens (Apparently not done by the Tracr solution)
                """
                outputSeq = []
                for t in tempSeq:
                    if t not in outputSeq:
                        for ii in range(tokenCounter[t]):
                            outputSeq.append(t)
                """

                inputSeq.insert(0, "BOS")
                outputSeq.insert(0, "BOS")

                data[i] = (inputSeq, outputSeq)

        case "shuffle_dyck1":
            acceptedTokens = acceptedNamesAndInput[name]

            for i in range(size):
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
                    if (
                        tokenDiff == 0
                    ):  # High probability of begining paranthesis if balanced
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

                data[i] = (inputSeq, outputSeq)

        case "shuffle_dyck2":
            acceptedTokens = acceptedNamesAndInput[name]

            for i in range(size):
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
                        split = (
                            1 - tokenProb[1]
                        )  # The reminder of probability to distribute
                        tokenProb[2] = split - split / (
                            (inputLength + 1)
                        )  # More likely to start a new parenthesis than break sequence
                        split = split - tokenProb[2]
                        tokenProb[0] = split / 2
                        tokenProb[3] = split / 2
                    elif tokenDiff2 > 0 and tokenDiff1 == 0:
                        tokenProb[3] = 1 - 1 / ((inputLength + 1) * tokenDiff2)
                        split = (
                            1 - tokenProb[3]
                        )  # The reminder of probability to distribute
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
                        split = (
                            1 - tokenProb[0]
                        )  # The reminder of probability to distribute
                        tokenProb[2] = split - split / (
                            (inputLength + 1)
                        )  # More likely to start a new parenthesis than break sequence
                        split = split - tokenProb[2]
                        tokenProb[1] = split / 2
                        tokenProb[3] = split / 2
                    elif tokenDiff2 < 0 and tokenDiff1 == 0:
                        tokenProb[2] = 1 - 1 / ((inputLength + 1) * (-tokenDiff2))
                        split = (
                            1 - tokenProb[2]
                        )  # The reminder of probability to distribute
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

                data[i] = (inputSeq, outputSeq)

        case _:
            print(
                name,
                "is not an accepted name the accepted names are",
                acceptedNamesAndInput,
            )
            return None

    if removeDuplicates:
        # Create a list of the unique found tokens

        # Try building up a dict list
        uniqueInputs = {}
        uniqueData = []

        # Might work, check with numpy
        # Check that numpy unique returns a jax array and not a numpy array
        for inputSeq, outputSeq in data:
            nextToken = uniqueInputs
            inputLength = len(inputSeq)
            for ind, token in enumerate(inputSeq):
                if token in nextToken.keys():
                    nextToken = nextToken[token]
                else:
                    nextToken[token] = {}
                    nextToken = nextToken[token]

            # Check if this sequence has been added to the sequence or not
            if "@end" not in nextToken.keys():
                nextToken["@end"] = {}
                uniqueData.append((inputSeq, outputSeq))

        data = uniqueData

    return data


# Generate a rasp model based on "name" and max sequence length "maxLength"
def generateModel(name: str, maxLength: int) -> Model:
    model = None
    acceptedNamesAndInput = getAcceptedNamesAndInput()

    match name:
        case "reverse":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(lib.make_reverse(rasp.tokens), inputs, maxLength, name)

        case "hist":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(lib.make_hist(), inputs, maxLength, name)

        case "sort":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(
                lib.make_sort(
                    rasp.tokens, rasp.tokens, max_seq_len=maxLength, min_key=min(inputs)
                ),
                inputs,
                maxLength,
                name,
            )

        case "most-freq":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(lib.make_sort_freq(maxLength), inputs, maxLength, name)

        case "shuffle_dyck1":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(lib.make_shuffle_dyck(["()"]), inputs, maxLength, name)

        case "shuffle_dyck2":
            inputs = {t for t in acceptedNamesAndInput[name]}
            model = Model(lib.make_shuffle_dyck(["()", "{}"]), inputs, maxLength, name)

        case _:
            print(
                name,
                "is not an accepted name the accepted names are",
                acceptedNamesAndInput,
            )
            return None

    return model


def make_sort_unique_buggy(vals: rasp.SOp, keys: rasp.SOp) -> rasp.SOp:
    """Returns vals sorted by < relation on keys.

    Only supports unique keys.

    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]

    Args:
      vals: Values to sort.
      keys: Keys for sorting.
    """
    # BUG: GT instead of LT, resulting in descending order instead of ascending
    smaller = rasp.Select(keys, keys, rasp.Comparison.GT).named("smaller")
    target_pos = rasp.SelectorWidth(smaller).named("target_pos")
    sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(sel_new, vals).named("sort")


def make_sort_buggy(
    vals: rasp.SOp, keys: rasp.SOp, *, max_seq_len: int, min_key: float
) -> rasp.SOp:
    """Returns vals sorted by < relation on keys, which don't need to be unique.

    The implementation differs from the RASP paper, as it avoids using
    compositions of selectors to break ties. Instead, it uses the arguments
    max_seq_len and min_key to ensure the keys are unique.

    Note that this approach only works for numerical keys.

    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens, 5, 1)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]
      sort([2, 4, 1, 2])
      >> [1, 2, 2, 4]

    Args:
      vals: Values to sort.
      keys: Keys for sorting.
      max_seq_len: Maximum sequence length (used to ensure keys are unique)
      min_key: Minimum key value (used to ensure keys are unique)

    Returns:
      Output SOp of sort program.
    """
    keys = rasp.SequenceMap(
        lambda x, i: x + min_key * i / max_seq_len, keys, rasp.indices
    )
    return make_sort_unique_buggy(vals, keys)


def generateBuggyModel(name: str, maxLength: int) -> Model:
    acceptedNamesAndInput = getAcceptedNamesAndInput()

    match name:
        case "sort":
            inputs = {t for t in acceptedNamesAndInput[name]}
            return Model(
                make_sort_buggy(
                    rasp.tokens, rasp.tokens, max_seq_len=maxLength, min_key=min(inputs)
                ),
                inputs,
                maxLength,
                name,
            )

        case _:
            print(
                name,
                "is not an accepted name the accepted names are",
                acceptedNamesAndInput,
            )
            return None


# Encode and the data for training based on the target function's rasp encoding
def encodeAndPadData(data, raspFunction: rasp.SOp, acceptedInputs, maxSeqLength: int):
    model = Model(raspFunction, acceptedInputs, maxSeqLength, "Temporary model")
    inputEncoder = model.model.input_encoder
    outputEncoder = model.model.output_encoder

    fillerToken = next(
        iter(outputEncoder.encoding_map)
    )  # A token accepted by the output encoder. Should not have an effect on loss

    X = []
    Y = []

    for inputSeq, outputSeq in data:
        # Copy sequences to avoid overwriting originals and pad them to max length
        x = []
        y = []
        for i in range(maxSeqLength + 1):
            if i < len(inputSeq):  # Assumes that input is same size as output
                x.append(inputSeq[i])
                y.append(outputSeq[i])
            else:
                x.append("compiler_pad")
                y.append(fillerToken)

        y[0] = fillerToken

        X.append(jnp.array(inputEncoder.encode(x)))
        Y.append(jnp.array(outputEncoder.encode(y)))

    X = jnp.array(X)
    Y = jnp.array(Y)

    return X, Y

    # NOTE The first token in the output is ignored but needs to be part of the input since the encoder breaks otherwise


def generateAndEncodeData(name: str, maxLength: int, size: int, removeDuplicates=True):
    data = generateData(name, maxLength, size, removeDuplicates)
    model = generateModel(name, maxLength)
    X, Y = encodeAndPadData(data, model.raspFunction, model.inputs, maxLength)
    return X, Y


# Prints some statistics on the generated dyck data
def checkDyckBalance(data):
    oddLength = 0
    balanced = 0

    for input, output in data:
        if len(input) % 2 == 0:  # length + bos
            oddLength += 1
        if output[1] == 1:
            balanced += 1

    oddLength /= len(data) / 100
    balanced /= len(data) / 100

    print("Percentage of data which is:")
    print("Of odd length:", oddLength)
    print("Balanced:", balanced)
