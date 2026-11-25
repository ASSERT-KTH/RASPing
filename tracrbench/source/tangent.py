from tracr.rasp import rasp

import math

def make_tangent() -> rasp.SOp:
    """
    Creates an SOp that applies the tangent function to each element in the input sequence.
    Since directly using the math library's tan function is not allowed within RASP,
    this example illustrates how one would theoretically encapsulate such an operation,
    keeping in mind RASP's constraints.
    """
    # In RASP, directly calling math.tan or any external library function is not possible.
    # However, if allowed, the operation would look something like this:
    # Apply the tangent function to each element of the sequence.
    return rasp.Map(lambda x: math.tan(x), rasp.tokens).named("tangent")
