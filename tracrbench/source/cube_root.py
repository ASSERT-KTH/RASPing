from tracr.rasp import rasp

def make_cube_root() -> rasp.SOp:
    """
    Creates an SOp that calculates the cube root of each element in the input sequence.

    Returns:
      A RASP SOp that when executed, calculates the cube root of each element in the input sequence.
    """
    # Define the cube root operation to be applied to each element.
    cube_root_operation = rasp.Map(lambda x: x**(1/3) if x >= 0 else -(-x)**(1/3), rasp.tokens).named("cube_root_operation")

    return cube_root_operation
