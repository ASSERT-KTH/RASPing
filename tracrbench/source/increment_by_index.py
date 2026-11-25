from tracr.rasp import rasp

def make_increment_by_index() -> rasp.SOp:
    # This operation adds each element of the input sequence to its corresponding index.
    incremented_sequence = rasp.SequenceMap(lambda x, y: x + y, rasp.tokens, rasp.indices).named("incremented_sequence")

    return incremented_sequence
