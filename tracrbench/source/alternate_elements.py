from tracr.rasp import rasp

def make_alternate_elements() -> rasp.SOp:
    return rasp.SequenceMap(lambda x,y: x if y%2==0 else y, rasp.tokens, rasp.indices)
