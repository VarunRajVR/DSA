class stack:
    def __init__(self):
        self.s = []

    def push(self,data):
        self.s.insert(0,data)
    def pop(self):
        if len(self.s) < 1:
            print("stack empty")
            return None
        return self.s.pop(0)
    def __str__(self) -> str:
        if not self.s:
            return "Queue is empty"
        out = ""
        i = self.s[0]
        for i in range(len(self.s)):
            out+=str(self.s[i]) + "->"
        return out
    
if __name__ == '__main__':
    s = stack()
    s.push(5)
    s.push(4)
    s.push(3)
    s.push(2)
    print(s)
    print(s.pop())
    print(s.pop())
    print(s.pop())
    print(s.pop())
    print(s.pop())
    