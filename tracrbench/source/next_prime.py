from tracr.rasp import rasp

def is_prime(num):
    """Check if a number is prime."""
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def next_prime(n):
    """Return the next highest prime number after n."""
    # Start checking from the next number
    prime_candidate = n
    while True:
        if is_prime(prime_candidate):
            return prime_candidate
        prime_candidate += 1


def make_next_prime():
    return  rasp.Map(next_prime, rasp.tokens)
