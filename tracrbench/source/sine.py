from tracr.rasp import rasp

import math

def make_sine() -> rasp.SOp:
    """
    Creates an SOp that applies the sine function to each element of the input sequence.

    Example usage:
      sine = make_sine()
      sine([0, math.pi/2, math.pi])
      >> [0, 1, 0]
    """
    # Apply the sine function to each element of the sequence
    return rasp.Map(lambda x: math.sin(x), rasp.tokens).named("sine")
