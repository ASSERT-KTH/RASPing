from tracr.rasp import rasp

def make_replace_small_tokens(threshold = 2) -> rasp.SOp:
    # Step 1: Create a comparison selector for elements smaller than the threshold.
    # This step involves directly applying the replacement logic as RASP does not support the initial planned approach.
    
    return rasp.Map(lambda x: 0 if x < threshold else x, rasp.tokens).named("replaced_sequence")
