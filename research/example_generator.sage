import random, sympy
from sage.all import factor, gcd

''' Helper function. Returns + or - 1 '''
def random_sign():
    x = random.randint(0,1)
    if x == 1:
        return 1
    else:
        return -1

''' Helper function. Generates q > p coprime '''

def gen_coprime(p, ratio=2):
    q = p
    while gcd(p,q)!=1:
        q = random.randint(p+1, 10*p)
    return q

''' Helper function. Generates coprime pair (p<q) '''

def gen_pair(size=100, ratio =10):
    p = random.randint(2,size)
    q = gen_coprime(p,ratio)
    return (p,q)

''' Helper function. For a given integer n returns a shuffled factorization of fixed length l'''

def refactor(n, l):
    factorized = list(factor(n))
    factors = []
    for x in factorized:
        for i in range(x[1]):
            factors.append(x[0])
    random.shuffle(factors)
    # Ensure we don't sample more than available, and handle small n
    k_val = max(0, l - 1)
    pop_size = max(0, len(factors) - 1)
    if k_val > pop_size:
        # If we need more factors than we have, we just take all of them as single partitions
        # This is a fallback for small n
        partitions = list(range(pop_size))
    else:
        partitions = random.sample(range(pop_size), k=k_val)
    
    partitions.sort()
    # Ensure the last element is the end of the list
    if not partitions or partitions[-1] != len(factors) - 1:
        partitions.append(len(factors) - 1)
    #print(partitions)
    result = []
    new_p = 1
    i = 0 
    for j in range(len(factors)):
        new_p *= factors[j]
        #print(j, i, new_p)
        if j == partitions[i]:
            result.append(new_p)
            new_p = 1
            i += 1
            #print("appending!")
    return result

''' Generates a random iterated torus knot of given length. with_sign adds the +1 sign and outputs the knot in the GA format '''

def gen_iterated(size=10, with_sign=False, size_range=100):
    k = []
    for i in range(size):
       k.append(gen_pair(size=size_range))
    
    if with_sign:
        return [(1, k)]
    else:
        return k

''' Generates a random connected sum '''

def gen_knot(size=10):
    result = []
    left = size
    while left > 0:
        localsize = random.randint(1,left)
        left -= localsize
        result.append( (random_sign(), gen_iterated(localsize)))
    return result

''' Kill first degree of the signature function as described in in the Lemma '''

def kill_first_degree(knot):
    killer = [ knot[0] ]
    
    P = 1
    for pq in knot[1:]:
        P*=pq[0]
    
    ps = refactor(P,len(knot)-1)

    for p in ps:
        killer.append((p, gen_coprime(p)))

    return killer


''' For a given iterated torus knot generates a knot J such that K#J is algebraically slice '''

# no error handling yet
def slicen(knot, sign=1):
    if len(knot) == 1:
        return [ (-sign, knot) ]

    result = []

    killer = kill_first_degree(knot)

    result.append((-sign, killer) )

    result+=(slicen(knot[1:], sign))
    result+=(slicen(killer[1:], -sign))

    return result
