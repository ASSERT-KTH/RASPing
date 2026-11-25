from tracr.rasp import rasp

def condition(a, idx1, idx2):
    if a==idx2:
        return idx1
    elif a==idx1:
        return idx2
    else:
        return a


def make_swap_elements(index_a = 1, index_b = 0):
    swaper = rasp.Map(lambda x: condition(x, index_a, index_b), rasp.indices)
    swap_selector = rasp.Select(swaper, rasp.indices, rasp.Comparison.EQ)
    #return swaper
    return rasp.Aggregate(swap_selector, rasp.tokens)
