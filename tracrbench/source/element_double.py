from tracr.rasp import rasp

def make_element_double() -> rasp.SOp:
    # Apply the doubling function to each element of the input sequence.
    return rasp.Map(lambda x: x * 2, rasp.tokens).named("double_elements")
