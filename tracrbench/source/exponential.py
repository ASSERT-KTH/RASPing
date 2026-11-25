from tracr.rasp import rasp

import math  # Importing math for the example, RASP's approximation will not use external libraries

def make_exponential() -> rasp.SOp:
    exp_approx = rasp.Map(lambda x: math.exp(x), rasp.tokens)
    return exp_approx
