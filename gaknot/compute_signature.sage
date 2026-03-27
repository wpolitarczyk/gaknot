pqs = [(6,5),(4,3)] #example

def a(p,q,x):
    for i in range(1,q):
        if (p*x*q-i*p)/q in ZZ:
            a=i
            b=(p*x*q-i*p)/q
            return a
    return 0
    
def b(p,q,x):
    for i in range(1,q):
        if (p*x*q-i*p)/q in ZZ:
            a=i
            b=(p*x*q-i*p)/q
            return b
    return 0

h = lambda p,q,x : (-1)^(math.floor(a(p,q,x)/q)+math.floor(b(p,q,x)/p)+math.floor(a(p,q,x)/q+b(p,q,x)/p))

f = lambda p,q,r,x : (p*q*x*r in ZZ and p*x*r not in ZZ and q*x*r not in ZZ)*h(p,q,x*r)

def fK (pqs , x):
    result = 0
    r=1
    for pq in pqs:
        p=pq[0]
        q=pq[1]
        result += f(p,q,r,x)
        r*=q
    return result

jumps = []
r=1

for j in pqs:
    p=j[0]
    q=j[1]

    xs = [i/(p*q*r) for i in range(1,p*q*r)]
    jumps += [x for x in xs if p*x not in ZZ and q*x not in ZZ]
    r*=q

jump_function_dict = {x: fK(pqs, x) for x in jumps}

