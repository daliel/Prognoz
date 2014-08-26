import math

PERCEPTRON():
    def __init__(self, ni, no):
        self.ni = ni
        self.no = no
    
    def makeMatrix(self, I, J, fill=1.0):
        m = []
        for i in range(I):
            m.append([fill]*J)
        return m
    
    def Sigmoid(self, x):
        a = math.e**(-x)
        b = 1+a
        c = 1/b
        return c
    
    def Train(self, puttern):
        if len(puttern) != self.ni: raise ValueError('wrong number of INPUTS values')