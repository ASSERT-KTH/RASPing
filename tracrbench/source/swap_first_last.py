from tracr.rasp import rasp

def condition(a,b):
    if a==0:
        return b-1
    elif a+1==b:
        return 0
    else:
        return a

def  make_swap_first_last():
    swaper = rasp.SequenceMap(lambda x,y: condition(x,y), rasp.indices, rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)))
    swap_selector = rasp.Select(swaper, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(swap_selector, rasp.tokens)
