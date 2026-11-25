from tracr.rasp import rasp

def shift_by(offset: int, sop: rasp.SOp) -> rasp.SOp:
    """Shifts the sequence by 'offset', filling the start with 'fill'."""
    select_off_by_offset = rasp.Select(rasp.indices, rasp.indices, lambda k, q: q == k + 1 or k==0 and q==0)
    return rasp.Aggregate(select_off_by_offset, sop)
    

def make_pairwise_max() -> rasp.SOp:
  
    
    # Shift the input sequence by 1 to the right, filling the first position with fill_value.
    shifted_sequence = shift_by(1, rasp.tokens).named("shifted_sequence")
    
    # Compare each element of the original sequence with the shifted sequence, taking the maximum.
    pairwise_max = rasp.SequenceMap(lambda x, y: int(max(x, y)), rasp.tokens, shifted_sequence).named("pairwise_max")
    
    return pairwise_max
