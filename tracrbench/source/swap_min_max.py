from tracr.rasp import rasp

def make_maximum() -> rasp.SOp:
    # Step 1: Create a comparison matrix with "Less Than or Equal" predicate
    unique_tokens = rasp.SequenceMap(lambda x,y: x+y*0.00001, rasp.tokens, rasp.indices)
    leq_selector = rasp.Select(unique_tokens, unique_tokens, rasp.Comparison.LEQ).named("leq_selector")

    # Step 2: Count the number of true comparisons for each element
    leq_count = rasp.SelectorWidth(leq_selector).named("leq_count")

    # Steps 3 & 4: Identify the sequence length to compare with the LEQ count
    all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
    sequence_length = rasp.SelectorWidth(all_true_selector).named("sequence_length")

    # Step 5: Identify the maximum element by its LEQ count matching the sequence length
    max_element_selector = rasp.Select(leq_count, sequence_length, rasp.Comparison.EQ).named("max_element_selector")

    # Step 6: Broadcast the maximum element across the entire output sequence
    max_sequence = rasp.Aggregate(max_element_selector, rasp.tokens).named("max_sequence")

    return max_sequence

def make_minimum() -> rasp.SOp:
    # Create a GEQ selector that compares each element with every other element.
    unique_tokens = rasp.SequenceMap(lambda x,y: x+y*0.00001, rasp.tokens, rasp.indices)
    geq_selector = rasp.Select(unique_tokens, unique_tokens, rasp.Comparison.GEQ).named("geq_selector")

    # Count the number of comparisons where each element is greater than or equal to other elements.
    geq_count = rasp.SelectorWidth(geq_selector).named("geq_count")

    # Create a sequence where each element is the length of the sequence to compare against.
    length = make_length()

    # Selector that selects the element where the geq_count equals the sequence length, identifying the minimum.
    min_element_selector = rasp.Select(geq_count, length, rasp.Comparison.EQ).named("min_element_selector")

    # Use Aggregate to broadcast the minimum element across the entire sequence.
    min_sequence = rasp.Aggregate(min_element_selector, rasp.tokens).named("min_sequence")

    return min_sequence

def make_length() -> rasp.SOp:
    all_true_selector = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
    return rasp.SelectorWidth(all_true_selector).named("length")



def make_swap_min_max() -> rasp.SOp:
    max_element = make_maximum()
    min_element = make_minimum()

    max_positions = rasp.SequenceMap(lambda x,y: y if x==y else -999999, rasp.tokens, max_element)
    min_positions = rasp.SequenceMap(lambda x,y: y if x==y else -999999, rasp.tokens, min_element)

    min_at_max_pos = rasp.SequenceMap(lambda x,y: y if x!=-999999 else -999999, max_positions, min_element)
    max_at_min_pos = rasp.SequenceMap(lambda x,y: y if x!=-999999 else -999999, min_positions, max_element)
    
    swapped_in_mins = rasp.SequenceMap(lambda x,y: x if y==-999999 else y, rasp.tokens, min_at_max_pos)
    swapped_sequence = rasp.SequenceMap(lambda x,y: x if y==-999999 else y, swapped_in_mins, max_at_min_pos)
    return swapped_sequence
