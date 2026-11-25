from tracr.rasp import rasp

def wrap_into_range(min, max, x):
    # Calculate the size of the range
    range_size = max - min 
    # Wrap x into the range
    wrapped_x = ((x - min) % range_size) + min
    return wrapped_x

def make_wrap(min_val = 2, max_val = 7) -> rasp.SOp:
    return rasp.Map(lambda x: wrap_into_range(min_val, max_val, x), rasp.tokens)
