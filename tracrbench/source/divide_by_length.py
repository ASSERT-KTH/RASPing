from tracr.rasp import rasp

  # Assuming RASP is already imported for the context of this task

def make_divide_by_length() -> rasp.SOp:
    # Step 1: Create a selector that selects all elements (TRUE condition)
    all_true_selector = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
    
    # Calculate the length of the sequence using SelectorWidth
    length = rasp.SelectorWidth(all_true_selector).named("length")
    
    # Step 2: Divide each element by the length of the sequence
    divided_by_length = rasp.SequenceMap(lambda x, length: x / length if length > 0 else 0, rasp.tokens, length).named("divided_by_length")
    
    return divided_by_length
