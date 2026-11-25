from tracr.rasp import rasp

def make_index_parity() -> rasp.SOp:
    parity_sequence = rasp.SequenceMap(lambda index, value: 0 if index % 2 == 0 else 1, rasp.indices, rasp.tokens)
    return parity_sequence
