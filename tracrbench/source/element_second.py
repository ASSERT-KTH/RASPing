from tracr.rasp import rasp

def make_element_second() -> rasp.SOp:
    # Select the second element by matching indices equal to 1.
    second_element_selector = rasp.Select(
        rasp.indices, # Keys: original indices of the sequence
        rasp.Map(lambda x: 1, rasp.indices), # Queries: creating a sequence of 1s to match the index 1
        rasp.Comparison.EQ # Predicate: equality check
    ).named("second_element_selector")

    # Use Aggregate to fill the sequence with the value of the second element.
    second_element_sequence = rasp.Aggregate(
        second_element_selector, # Selector that identifies the second element
        rasp.tokens, # SOp: the input sequence
        # Note: default is None as per task rules
    ).named("second_element_sequence")

    return second_element_sequence
