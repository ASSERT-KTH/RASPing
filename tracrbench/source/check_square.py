from tracr.rasp import rasp

def make_check_square() -> rasp.SOp:
    return rasp.Map(lambda x: 1 if x**0.5 == int(x**0.5) else 0, rasp.tokens)
