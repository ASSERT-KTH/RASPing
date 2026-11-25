from tracr.rasp import rasp

def make_first_element() -> rasp.SOp:
    # Selector that identifies the first element by comparing indices to 0.
    first_elem_selector = rasp.Select(rasp.indices, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ).named("first_elem_selector")
    
    # Use Aggregate to broadcast the first element across the entire sequence.
    first_element_sequence = rasp.Aggregate(first_elem_selector, rasp.tokens, default=None).named("first_element_sequence")
    
    return first_element_sequence
