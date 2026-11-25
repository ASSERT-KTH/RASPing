from tracr.rasp import rasp

def make_rotate_left():
    len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
    rotator = rasp.SequenceMap(lambda x,y: ((x+1)%(y)) if y!=0 else 0, rasp.indices, len)
    rotate_selector = rasp.Select(rasp.indices, rotator, rasp.Comparison.EQ)
    return rasp.Aggregate(rotate_selector, rasp.tokens)
