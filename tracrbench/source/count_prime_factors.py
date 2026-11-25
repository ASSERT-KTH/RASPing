from tracr.rasp import rasp

def count_prime_factors(n):
    if n==0: return 0
    """Count the distinct prime factors of a number."""
    count = 0
    # Handle 2 separately to make the loop only for odd numbers
    if n % 2 == 0:
        count += 1
        while n % 2 == 0:
            n //= 2
    # Check for odd factors
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            count += 1
            while n % factor == 0:
                n //= factor
        factor += 2
    # If n is a prime number greater than 2
    if n > 2:
        count += 1
    return count

def make_count_prime_factors():
    return rasp.Map(count_prime_factors, rasp.tokens)
