from tracr.rasp import rasp

def make_negation() -> rasp.SOp:
    """
    Creates a RASP SOp that negates each element in the input sequence.
    
    Returns:
        A RASP SOp that, when executed, produces a sequence where each element
        is the negation of the corresponding element in the input sequence.
    """
    # Define the negation function
    negation_function = lambda x: -x
    
    # Apply the negation function element-wise to the input sequence
    negated_sequence = rasp.Map(negation_function, rasp.tokens).named("negated_sequence")
    
    return negated_sequence
