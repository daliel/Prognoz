import math
from decimal import Decimal
def main(x):
    ac= Decimal(1 /(1+Decimal(math.e)**(Decimal(-x))))
    print ac
    #print 1+math.e**-x
    #print math.log(1-ac, math.e)
    demain(ac)
def demain(y):
    a =Decimal(1/y)
    b = Decimal(a-1)
    print math.fabs(math.log(b)), "Math"
    print math.fabs(b.ln()), "Decimal"
    

main(61)