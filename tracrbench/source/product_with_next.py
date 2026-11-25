from tracr.rasp import rasp

def make_product_with_next() -> rasp.SOp:
    # Function to shift the sequence by one position
    def shift_by_one() -> rasp.SOp:
        # Define a selector for shifting sequence by one
        len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)) 
        shifted_selector = rasp.Select(rasp.indices, rasp.SequenceMap(lambda x,y: x if x<y-1 else x-1,rasp.indices, len), lambda x,y: x-1==y)
        return rasp.Aggregate(shifted_selector, rasp.tokens)

    shifted_sequence = shift_by_one()
    # Add the original sequence to the shifted sequence
    sum_with_next = rasp.SequenceMap(lambda x, y: x * y, rasp.tokens, shifted_sequence)

    return sum_with_next
