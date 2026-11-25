from tracr.rasp import rasp

def make_increment() -> rasp.SOp:
    # Define a Map operation to increment each element in the sequence by 1.
    return rasp.Map(lambda x: x + 1, rasp.tokens).named("increment")
