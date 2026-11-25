from tracr.rasp import rasp

def make_sign() -> rasp.SOp:
    """
    Creates an SOp that assigns -1, 0, or 1 to each element of the input sequence
    based on its sign (negative, zero, or positive, respectively).
    """
    # Define the function to apply to each element
    def sign_check(x):
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return 0

    # Apply the sign checking function to each element of the input sequence
    return rasp.Map(sign_check, rasp.tokens).named("sign")
