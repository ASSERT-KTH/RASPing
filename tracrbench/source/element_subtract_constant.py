from tracr.rasp import rasp

def make_element_subtract_constant(constant = 2) -> rasp.SOp:

    subtract_constant = rasp.Map(lambda x: x - constant, rasp.tokens).named(f"subtract_{constant}")
    
    return subtract_constant
