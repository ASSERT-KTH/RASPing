from tracr.rasp import rasp

def make_cube_each_element() -> rasp.SOp:
    """
    Returns a RASP SOp that when executed, cubes each element of the input sequence.
    """
    # Apply the cubing function to each element of the input sequence.
    cube_sequence = rasp.Map(lambda x: x ** 3, rasp.tokens).named("cube_sequence")

    return cube_sequence
