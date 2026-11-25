from tracr.rasp import rasp

def make_swap_odd_index() -> rasp.SOp:
    len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
    swaper = rasp.SequenceMap(lambda x, y: x if (x==y-1 and y%2==0 or x==0) else (x-1 if (x+1)%2==1 else x+1), rasp.indices, len)
    swap_selector = rasp.Select(rasp.indices, swaper, rasp.Comparison.EQ)
    swaped = rasp.Aggregate(swap_selector, rasp.tokens)
    return swaped
