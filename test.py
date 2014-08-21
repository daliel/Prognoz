import math
def main(x):
    ac= 1/(1+math.e**-x)
    print ac
    #print 1+math.e**-x
    #print math.log(1-ac, math.e)
    demain(ac)
def demain(ac):
    n = ((1/ac)-1)* math.e
    print math.log((1/ac)-1)
    """for i in xrange(1,53):
        if n == float(-1):
            print n, i
            return
        n= n * math.e
        
        print n, type(n)"""

main(0.1)