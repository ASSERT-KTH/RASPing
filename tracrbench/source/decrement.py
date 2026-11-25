from tracr.rasp import rasp

def make_decrement() -> rasp.SOp:
    # The Map function is used to decrement each element by 1.
    # The lambda function specifies the operation to perform on each element of the input sequence.
    return rasp.Map(lambda x: x - 1, rasp.tokens).named("decrement")
