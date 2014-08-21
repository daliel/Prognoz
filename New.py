from Tkinter import *

import ttk
import math

class APP():
    def __init__(self):
        self.root = Tk()
        self.mainValues = [u"NN with one hiden layer and output ~ 50"]
        self.combobox = ttk.Combobox(self.root, values = self.mainValues,height=3, width = 50, state="readonly")
        self.combobox.set(self.mainValues[0])
        self.combobox.grid()
        self.mainButton = Button(self.root, text = "Next", command = self.MainButton)
        self.mainButton.grid()
        self.run()
    def MainButton(self):
        for i in xrange(len(self.mainValues)):
            if self.mainValues[i] == self.combobox.get():
                if i == 0:
                    print self.mainValues[i]
    def run(self):
        self.root.mainloop()
        
class NNBP():
    def __init__(self, ni, nh, no):   
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = self.makeMatrix(self.ni, self.nh)
        self.wo = self.makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = 1
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = 1

        # last change in weights for momentum   
        self.ci = self.makeMatrix(self.ni, self.nh)
        self.co = self.makeMatrix(self.nh, self.no)
    
    # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
    def Activation(self, x):
        return 1 /(1+(math.e**-x))
        
    # Make a matrix (we could use NumPy to speed this up)
    def makeMatrix(self, I, J, fill=0.0):
        m = []
        for i in range(I):
            m.append([fill]*J)
        return m
    
    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def DeActivation(self, y):
        n = ( 1-(1/y))*math.e
        for i in xrange(1, 53):
            if n== -1.0: return i
            n*=math.e
        
        return 0
    
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
        f = open(fname, "r")
        line = f.read()
        f.close()
        arr = string.split(line, ";")
        self.ni, self.nh, self.no = int(arr[0]), int(arr[1]), int(arr[2])
        self.wi = self.makeMatrix(self.ni, self.nh)
        self.wo = self.makeMatrix(self.nh, self.no)
        n = 2
        for i in xrange(self.ni):
            for j in xrange(self.nh):
                n += 1
                self.wi[i][j] = float(arr[n])
        for i in xrange(self.nh):
            for j in xrange(self.no):
                n += 1
                self.wo[i][j] = float(arr[n])
        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
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
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = self.Activation(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = self.Activation(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = self.DeActivation(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = self.DeActivation(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                #print "len(self.co)", len(self.co), len(self.co[j])
                #print "len(self.wo)", len(self.wo), len(self.wo[j])
                #print "j, k", j, k
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
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
        error = 999999999999.0
        i = 0
        while error>len(patterns) or not self.st:
            error = 0.0
            for p in patterns:
                print p[0]
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            self.save("%s.sav"%self.nh)
            #self.write_log(i, error)
            i+=1
           
        self.terminate()
        
if __name__ == "__main__":
    nn = NNBP(6,1,52)
    a = nn.Activation(1)
    print a, "a"
    print nn.DeActivation(a)