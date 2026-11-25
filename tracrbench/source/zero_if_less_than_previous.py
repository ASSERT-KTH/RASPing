from tracr.rasp import rasp

def make_zero_if_less_than_previous() -> rasp.SOp:
    # Correctly shift the sequence by using Aggregate and Select to create the shifted sequence with the placeholder at the start
    shifted_sequence = rasp.Aggregate(
        rasp.Select(rasp.indices, rasp.indices, lambda k, q: q == k + 1 or k==0 and q==0),
        rasp.tokens
    ).named("shifted_sequence_with_placeholder")

    
    # Use SequenceMap to compare each element with its shifted version
    zero_if_less_than_previous = rasp.SequenceMap(
        lambda original, shifted: 0 if original < shifted else original,
        rasp.tokens,
        shifted_sequence
    ).named("zero_if_less_than_previous")

    return zero_if_less_than_previous
