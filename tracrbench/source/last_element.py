from tracr.rasp import rasp

def make_last_element() -> rasp.SOp:
    # Generating the length of the sequence
    length = rasp.SelectorWidth(rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)).named("length")
    
    # Selector for the last element based on length minus 1
    last_element_selector = rasp.Select(rasp.indices, rasp.Map(lambda x: x-1, length), rasp.Comparison.EQ).named("last_element_selector")
    
    # Broadcasting the last element across the entire sequence
    last_element_sequence = rasp.Aggregate(last_element_selector, rasp.tokens).named("last_element_sequence")
    
    return rasp.SequenceMap(lambda x,y: x if y==0 else 0, last_element_sequence, rasp.indices)
