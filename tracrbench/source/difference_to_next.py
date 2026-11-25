from tracr.rasp import rasp

def  make_difference_to_next():
    def shift_by_one() -> rasp.SOp:
        # Define a selector for shifting sequence by one
        len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)) 
        shifted_selector = rasp.Select(rasp.indices, rasp.SequenceMap(lambda x,y: x if x<y-1 else x-1,rasp.indices, len), lambda x,y: x-1==y)
        return rasp.Aggregate(shifted_selector, rasp.tokens)

    shifted = shift_by_one()
    return rasp.SequenceMap(lambda x,y: x-y, shifted, rasp.tokens)
