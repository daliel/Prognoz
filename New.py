from Tkinter import *
from tkFileDialog import *
from decimal import Decimal

import ttk
import math
import other

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
                    self.BackPropagate()
                    
    def BackPropagate(self):        
        self.combobox.grid_forget()
        self.mainButton.grid_forget()
        self.root.update()
        #self.NNBP_New_Teach = Button(self.root, text = "New Teach", command = self.)
        options = {"title": "Open Data File", "filetypes": [('text files', '.txt')]}
        dataFile = askopenfilename(**options)
        putterns = other.create_pattern_52(other.read_file_data_r(dataFile))
        nn = NNBP(6,6, 52)
        nn.load("%i.sav"%nn.nh)
        nn.train(putterns)
        self.root.close()
    
    def run(self):
        self.root.mainloop()
        
class NNBP():
    def __init__(self, ni, nh, no):   
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [Decimal(1.0)]*self.ni
        self.ah = [Decimal(1.0)]*self.nh
        self.ao = [Decimal(1.0)]*self.no
        
        # create weights
        self.wi = self.makeMatrix(self.ni, self.nh)
        self.wo = self.makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = Decimal(1)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = Decimal(1)

        # last change in weights for momentum   
        self.ci = self.makeMatrix(self.ni, self.nh)
        self.co = self.makeMatrix(self.nh, self.no)
    
    # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
    def Activation(self, x):
        return Decimal(1 /(1+Decimal(math.e)**(Decimal(-x))))
        
    # Make a matrix (we could use NumPy to speed this up)
    def makeMatrix(self, I, J, fill=0.0):
        m = []
        fill= Decimal(fill)
        for i in range(I):
            m.append([fill]*J)
        return m
    
    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def DeActivation(self, y):
        if y == 1: return 0
        a =Decimal(1/y)
        b = Decimal(a-1)
        return math.fabs(b.ln())
    
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
        print fname
        f = open(fname, "r")
        line = f.read()
        f.close()
        arr = line.split( ";")
        self.ni, self.nh, self.no = int(arr[0]), int(arr[1]), int(arr[2])
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
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = self.Activation(sum)

        # output activations
        for k in range(self.no):
            sum = Decimal(0.0)
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
            error = Decimal(targets[k]-self.ao[k])
            output_deltas[k] = Decimal(self.DeActivation(self.ao[k])) * Decimal(error)

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = Decimal(0.0)
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            if Decimal(self.DeActivation(self.ah[j]))<= -999999999 or Decimal(self.DeActivation(self.ah[j]))>= 999999999:
                print self.ah[j], "self.ah[j]"
                a =Decimal(1/self.ah[j])
                print a, "1/self.ah[j]"
                b = Decimal(a-1)
                print b, "1/self.ah[j]-1"
                print b.ln(), "b.ln()"
            hidden_deltas[j] = Decimal(self.DeActivation(self.ah[j])) * error
        
        # calculate error
        error = Decimal(0.0)
        for k in range(len(targets)):
            error = error + Decimal(targets[k]-self.ao[k])
        
            
        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                #print "len(self.co)", len(self.co), len(self.co[j])
                #print "len(self.wo)", len(self.wo), len(self.wo[j])
                #print "j, k", j, k
                self.wo[j][k] = self.wo[j][k] + Decimal(N)*change + Decimal(M)*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + Decimal(N)*change + Decimal(M)*self.ci[i][j]
                self.ci[i][j] = change

        
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
        i = 0
        while (math.fabs(error))>(len(patterns)*2):
            if error != 0.0: N = math.fabs(error)/len(patterns)
            error = Decimal(0.0)
            print N, "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"
            for p in patterns:
                print p[0]
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
                print error, "Error"
            self.save("%s.sav"%self.nh)
        print math.fabs(error)
           
        
        
if __name__ == "__main__":
    APP()