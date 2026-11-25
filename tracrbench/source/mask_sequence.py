from tracr.rasp import rasp

def make_mask_sequence(index = 1) -> rasp.SOp:
    return rasp.SequenceMap(lambda x, y: x if y==index else 0, rasp.tokens, rasp.indices)
