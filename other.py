
def TransToIntList(e):
    a = []
    if len(e)<52:
        return -1
    for i in range(52):
        if e[i] == 1:
            a.append(i+1)
    return a
    
def IntListToTrans( e):
    a = [0 for i in xrange(52)]
    if type(e) == list:
        for j in xrange(len(e)):
            a[e[j]-1]=1
    else:
        a[e-1]=1
    return a[:] 
    


def read_file_data(fname):
    result = []
    fr = open(fname, "r")
    buf = fr.readlines()
    fr.close()
    i = 0
    while i<len(buf):
        temp = buf[i].split(" ")
        temp1 = []
        
        for n in range(4,10):
            
            temp1.append(int(temp[n]))
        result.append(temp1)
        i+=1
    return result
    
def read_file_data_r(fname):
    result = []
    fr = open(fname, "r")
    buf = fr.readlines()
    fr.close()
    i = len(buf)-1
    #print i
    while i>-1:
        temp = buf[i].split("\t")
        temp1 = []
        #print buf[i]
        for n in range(4,10):
            #print int(temp[n])
            temp1.append(int(temp[n]))
        result.append(temp1)
        i-=1
    return result
    
def demo():
    # Teach network XOR function
   
    tr = read_file_data("data.txt")
    p=create_pattern(tr)
    # create a network with two input, two hidden, and one output nodes
    for i in xrange(1, 52*52):
        n = NN(6, i, 52)
        # train it with some patterns
        print n.ni, n.nh, n.no, i
        n.train(patterns = p, iterations = i*1000)
        # test it
        n.test(p)
        n.save("%i_z.txt"%i)
        #n.load("z.txt")
        n.test(p)

    
if __name__ == "__main__":
    read_file_data("data.txt")
    
    
    
    
    
    
    
    
    
    
    
    
    
    