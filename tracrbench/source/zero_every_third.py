from tracr.rasp import rasp

def make_zero_every_third() -> rasp.SOp:
    # Step 1: Use rasp.indices to generate a sequence of indices.
    
    # Step 2: Map over the indices to identify every third element, considering 1-based indexing.
    every_third = rasp.Map(lambda x: (x + 1) % 3 == 0, rasp. indices).named("every_third")
    
    # Step 3: Convert boolean flags (True/False) to 0/1 for easier handling in SequenceMap.
    every_third_numerical = rasp.Map(lambda x: 1 if x else 0, every_third).named("every_third_numerical")
    
    # Step 4: Use SequenceMap to set every third element to 0 and leave others unchanged.
    result_sequence = rasp.SequenceMap(lambda x, is_third: 0 if is_third else x, rasp.tokens, every_third_numerical).named("result_sequence")
    
    return result_sequence
