from tracr.rasp import rasp

def make_square_each_element() -> rasp.SOp:
    # Applying the squaring function element-wise on the input sequence.
    return rasp.Map(lambda x: x**2, rasp.tokens).named("square_each_element")
