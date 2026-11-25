from tracr.rasp import rasp

def  make_interleave_reverse():
    len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
    shifter = rasp.SequenceMap(lambda x,y: x if x%2==0 else (y-x if y%2==0 else y-x-1), rasp.indices, len)
    shift_selector = rasp.Select(rasp.indices,shifter, rasp.Comparison.EQ)
    return rasp.Aggregate(shift_selector, rasp.tokens)
