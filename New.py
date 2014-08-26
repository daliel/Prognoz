from Tkinter import *
from tkFileDialog import *

import ttk
import other
import os

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
        putterns = other.create_pattern_6(other.read_file_data_r(dataFile))
        nn = NNBP(6,6, 6)
        
        nn.load("%i.sav"%nn.nh)
        nn.weights()
        nn.train(putterns)
        self.root.close()
    
    def run(self):
        self.root.mainloop()

           
        
        
if __name__ == "__main__":
    APP()