from tracr.rasp import rasp

def make_check_multiple_of_first():
    first = rasp.Aggregate(rasp.Select(rasp.indices, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ), rasp.tokens)
    return rasp.SequenceMap(lambda x,y: (1 if x%y==0 else 0) if y!=0 else 0, rasp.tokens, first)
