from decimal import Decimal
import math

        
class NNBP():
    def __init__(self, ni, nh, no):   
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [Decimal(0.5)]*self.ni
        self.ah = [Decimal(0.5)]*self.nh
        self.ao = [Decimal(0.5)]*self.no
        
        # create weights
        self.wi = self.makeMatrix(self.ni, self.nh)
        self.wo = self.makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = Decimal(0.5)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = Decimal(0.5)

        # last change in weights for momentum   
        self.ci = self.makeMatrix(self.ni, self.nh)
        self.co = self.makeMatrix(self.nh, self.no)
    
    # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
    def Activation(self, x):
        a = Decimal(math.e)**Decimal(-x)
        b = 1+a
        c = 1/b
        return c
        
    # Make a matrix (we could use NumPy to speed this up)
    def makeMatrix(self, I, J, fill=1.0):
        m = []
        fill= Decimal(fill)
        for i in range(I):
            m.append([fill]*J)
        return m
    
    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def DeActivation(self, y):
        if y == 1: return 1
        a =Decimal(1/y)
        b = Decimal(a-1)
        return 0-(b.ln())
    
    # calculate a random number where:  a <= rand < b
    def rand(self, a, b):
        return (b-a)*random.random() + a

    
    def save(self, fname):
        f = open(fname, "w")
        f.write(str(self.ni)+";")
        f.write(str(self.nh)+";")
        f.write(str(self.no)+";")
        for i in xrange(self.ni):
            for j in xrange(self.nh):
                f.write(str(self.wi[i][j])+";")
        for i in xrange(self.nh):
            for j in xrange(self.no):
                f.write(str(self.wo[i][j])+";")
        f.close()

    def load(self, fname):
        try: 
            f = open(fname, "r")
            line = f.read()
            f.close()
        
            arr = line.split( ";")
            self.ni, self.nh, self.no = int(arr[0]), int(arr[1]), int(arr[2])
        except: return
        self.wi = self.makeMatrix(self.ni, self.nh)
        self.wo = self.makeMatrix(self.nh, self.no)
        n = 2
        for i in xrange(self.ni):
            for j in xrange(self.nh):
                n += 1
                self.wi[i][j] = Decimal(arr[n])
        for i in xrange(self.nh):
            for j in xrange(self.no):
                n += 1
                self.wo[i][j] = Decimal(arr[n])
        # activations for nodes
        self.ai = [Decimal(1.0)]*self.ni
        self.ah = [Decimal(1.0)]*self.nh
        self.ao = [Decimal(1.0)]*self.no
        self.ci = self.makeMatrix(self.ni, self.nh)
        self.co = self.makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = Decimal(0.0)
            for i in range(self.ni):
                sum = sum + (self.ai[i] * self.wi[i][j])
            self.ah[j] = self.Activation(sum)

        # output activations
        for k in range(self.no):
            sum = Decimal(0.0)
            for j in range(self.nh):
                sum = sum + (self.ah[j] * self.wo[j][k])
            self.ao[k] = self.Activation(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = Decimal(targets[k]-self.ao[k])
            #print targets[k],  "targets[k]"
            #print error, "error output delta"
            output_deltas[k] = Decimal(self.DeActivation(self.ao[k])) * Decimal(error)

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = Decimal(0.0)
            for k in range(self.no):
                error = error + (output_deltas[k]*self.wo[j][k])
            if Decimal(self.DeActivation(self.ah[j]))<= -999999999 or Decimal(self.DeActivation(self.ah[j]))>= 999999999:
                print self.ah[j], "self.ah[j]"
                a =Decimal(1/self.ah[j])
                print a, "1/self.ah[j]"
                b = Decimal(a-1)
                print b, "1/self.ah[j]-1"
                print b.ln(), "b.ln()"
            hidden_deltas[j] = Decimal(self.DeActivation(self.ah[j])) * error        
            
        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                #print change, "change"
                #print "len(self.co)", len(self.co), len(self.co[j])
                #print "len(self.wo)", len(self.wo), len(self.wo[j])
                #print "j, k", j, k
                self.wo[j][k] -= (N*change)
                """self.wo[j][k] = self.wo[j][k] - (Decimal(N)*change) + (Decimal(M)*self.co[j][k])"""
                self.co[j][k] = change
                #rint (Decimal(N)*change), (Decimal(M)*self.co[j][k])

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[j][k] -= (N*change)
                #self.wi[i][j] = self.wi[i][j] + (Decimal(N)*change) + (Decimal(M)*self.ci[i][j])
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + targets[k]-float(self.ao[k])
        
        return error


    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def run(self):
        self.train(self.putterns)
        
          
    
    def train(self, patterns, N=1, M=0.1):
        # N: learning rate
        # M: momentum factor
        error = (len(patterns)*2)+1
        lastError = 0.0
        i = 0
        while (math.fabs(error))>(len(patterns)*2):
            #if error != 0.0: N = error/52
            print lastError, type(lastError),  error,type(error), int(lastError), int(error)
            if int(lastError) == int(error): 
                N+=1
            lastError = error
            print error, "ERROR"
            error = 0.0
            print N, "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"
            for p in patterns:
                #print p[0]
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
                #print error, "Error"
            self.save("%s.sav"%self.nh)
        print math.fabs(error)
        
if __name__ = "__main__":
    NNBP(6,6,6)