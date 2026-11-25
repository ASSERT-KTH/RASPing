from tracr.rasp import rasp

def make_sum_of_last_two() -> rasp.SOp:
    len = rasp.SelectorWidth(rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE))
    last_idx = rasp.Map(lambda x: x-1, len)
    second_to_last_idx = rasp.Map(lambda x: x-2, len)
    last_elt = rasp.Aggregate(rasp.Select(rasp.indices, last_idx, rasp.Comparison.EQ), rasp.tokens)
    second_to_last_elt = rasp.Aggregate(rasp.Select(rasp.indices, second_to_last_idx, rasp.Comparison.EQ), rasp.tokens)
    return rasp.SequenceMap(lambda x,y: x+y, last_elt, second_to_last_elt)
