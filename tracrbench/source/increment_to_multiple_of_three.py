from tracr.rasp import rasp

def make_increment_to_multiple_of_three() -> rasp.SOp:
    # Map operation to increment each element until it becomes a multiple of 3
    increment_to_multiple_of_three = rasp.Map(lambda x: x + (3 - (x % 3)) % 3, rasp.tokens).named("increment_to_multiple_of_three")
    
    return increment_to_multiple_of_three
