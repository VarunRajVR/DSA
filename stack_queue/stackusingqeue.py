class DEqueue():
    def __init__(self):
        self.l = []

    def enq_end(self, data):
        self.l.append(data)

    def enq_beg(self, data):
        self.l.insert(0,data)

    def deq_end(self):
        if self.l:
            return self.l.pop()  
        else:
            return None  

    def deq_beg(self):
        if self.l:
            return self.l.pop(0)  
        else:
            return None  
        #display the function
    def __str__(self) -> str:
        if not self.l:
            return "Queue is empty"
        out = "->".join(str(item) for item in self.l)
        return out


class stack():
    def __init__(self):
        self.s = DEqueue()

    def push(self,data):
        self.s.enq_beg(data)

    def pop(self):
        result = self.s.deq_beg()
        if result is not None:
            print(result)
    
    def __str__(self) -> str:
        return str(self.s) 
            
if __name__ == '__main__':
    varun = stack()
    varun.push(9)
    varun.push(8)

    print(varun)

    varun.pop()
    varun.pop()
    print(varun)