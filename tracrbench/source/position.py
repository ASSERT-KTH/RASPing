from tracr.rasp import rasp

def make_position() -> rasp.SOp:
    one_based_positions = rasp.SequenceMap(lambda x,y: x + 1, rasp.indices, rasp.tokens)
    return one_based_positions
