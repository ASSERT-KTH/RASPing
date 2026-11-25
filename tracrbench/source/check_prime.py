from tracr.rasp import rasp


def primecheck(n):
    if n < 2:
        return 0
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return 0
    return 1

def make_check_prime() -> rasp.SOp:
    return rasp.Map(lambda x: primecheck(x), rasp.tokens)
