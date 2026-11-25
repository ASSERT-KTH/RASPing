from tracr.rasp import rasp

def make_flip_halves() -> rasp.SOp:
    len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
    half = rasp.Map(lambda x: x/2, len)
    new_positions = rasp.SequenceMap(lambda x,y: (x-y if x>=y else x+y) if y==int(y) else (x if x+0.5==y else (x+int(y)+1 if x<y else x-int(y)-1)), rasp.indices, half)
    shifter = rasp.Select(new_positions, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(shifter, rasp.tokens)
