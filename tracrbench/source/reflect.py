from tracr.rasp import rasp

def reflect_into_rage(max,min, x):
    d = max-min
    if x> min and x<max:
        return x
    elif x<min:
        delta = min-x
        i = (delta//d)%2
        if i == 0:
            return min + (delta % d)
        else:
            return max - (delta % d)
    else:
        delta = x-max
        i = (delta//d)%2
        if i == 1:
            return min + (delta % d)
        else:
            return max - (delta % d)

def make_reflect(min_val = 2, max_val = 7) -> rasp.SOp:
    return rasp.Map(lambda x: reflect_into_rage(max_val, min_val, x), rasp.tokens)
