from tracr.rasp import rasp

import math

def make_cosine() -> rasp.SOp:
    """
    Creates an SOp that applies the cosine function to each element of the input sequence.

    Returns:
        A RASP SOp that when executed, applies cosine to each element of the input sequence.
    """
    # Apply the cosine function element-wise.
    # It's assumed here that we're able to use Python's math.cos. In a more restricted RASP environment,
    # an alternative method of computing the cosine might be necessary.
    return rasp.Map(lambda x: math.cos(x), rasp.tokens).named("apply_cosine")
