from tracr.rasp import rasp

def make_check_fibonacci() -> rasp.SOp:
    # Assume a pre-generated Fibonacci sequence up to a certain limit.
    # In practice, this would need to be dynamically generated or sufficiently large.
    fib_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    
    # Function to check if a number is in the Fibonacci sequence
    is_fib = lambda x: 1 if x in fib_sequence else 0

    # Apply the check to each element of the input sequence
    check_fibonacci_map = rasp.Map(is_fib, rasp.tokens).named("check_fibonacci_map")
    
    return check_fibonacci_map
