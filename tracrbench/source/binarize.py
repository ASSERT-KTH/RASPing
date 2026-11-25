from tracr.rasp import rasp

def make_binarize(threshold=3) -> rasp.SOp:
    compare_to_threshold = rasp.Map(lambda x: x >= threshold, rasp.tokens)
    binarized_sequence = rasp.Map(lambda x: 1 if x else 0, compare_to_threshold)
    
    return binarized_sequence
