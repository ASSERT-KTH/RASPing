from tracr.rasp import rasp

def make_decrement_to_multiple_of_three() -> rasp.SOp:
    """
    Creates an SOp that decrements each element in the sequence until it becomes a multiple of 3.
    """
    return rasp.Map(lambda x: x - x % 3, rasp.tokens).named("decrement_to_multiple_of_three")
