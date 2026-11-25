from tracr.rasp import rasp

def  make_check_divisibility(divisor = 3):
    return rasp.Map(lambda x: 1 if x%divisor == 0 else 0, rasp.tokens)
