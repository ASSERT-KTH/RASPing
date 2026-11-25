from tracr.rasp import rasp

def  make_set_to_index() -> rasp.SOp:
    one_based_positions = rasp.SequenceMap(lambda x,y: x, rasp.indices, rasp.tokens)
    return one_based_positions
