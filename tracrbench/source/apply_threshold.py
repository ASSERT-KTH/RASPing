from tracr.rasp import rasp

def make_apply_threshold(threshold = 3) -> rasp.SOp:
    
    apply_threshold_operation = rasp.Map(lambda x: 0 if x < threshold else x, rasp.tokens).named("apply_threshold")

    return apply_threshold_operation
