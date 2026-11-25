from tracr.rasp import rasp

def make_element_divide() -> rasp.SOp:
    # Step 1: Select the first element
    first_elem_selector = rasp.Select(
        rasp.indices,
        rasp.Map(lambda x: 0, rasp.indices),
        rasp.Comparison.EQ
    ).named("first_elem_selector")
    first_elem = rasp.Aggregate(first_elem_selector, rasp.tokens).named("first_elem")
    len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
    # Step 2: Select the second element
    second_elem_selector = rasp.Select(
        rasp.indices,
        rasp.Map(lambda x: 1 if x>1 else 0,len),
        rasp.Comparison.EQ
    ).named("second_elem_selector")
    second_elem = rasp.Aggregate(second_elem_selector, rasp.tokens).named("second_elem")


    # Step 3: Divide the second element by the first to get the divisor
    divisor = rasp.SequenceMap(lambda x, y: y / x if x != 0 and y != 0 else 1, first_elem, second_elem).named("divisor")

    # Step 4: Divide each element of the input sequence by the divisor
    result_sequence = rasp.SequenceMap(lambda x, y: x / y, rasp.tokens, divisor).named("result_sequence")

    return result_sequence
