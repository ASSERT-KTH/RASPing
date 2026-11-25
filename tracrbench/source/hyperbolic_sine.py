from tracr.rasp import rasp

import math

def make_hyperbolic_sine() -> rasp.SOp:
    """
    Creates an SOp that applies the hyperbolic sine function to each element of the input sequence.

    Returns:
        A RASP SOp that, when executed, applies math.sinh to each element of the input sequence.
    """
    # Apply math.sinh to each element in the sequence.
    hyperbolic_sine = rasp.Map(lambda x: math.sinh(x), rasp.tokens).named("hyperbolic_sine")

    return hyperbolic_sine
