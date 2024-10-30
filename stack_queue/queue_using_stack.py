class stack():
    def __init__(self) -> None:
        self.s = []
    def push(self, data):
        self.s.append(data)

    def pop(self):
        if self.s:
            return self.s.pop()
        else: return None

    def __str__(self) -> str:
        out=""
        if self.s:
            out =  ",".join(str(i) for i in self.s)
            return out
        else:
            return "stack empty"
        
class Queue():
    def __init__(self) -> None:
        self.s1 = stack()
        self.s2 = stack()
    def enqueue(self, data):
        self.s1.push(data)
    def dequeue(self):
        if not self.s2.s:  # Check if s2 is empty
            while self.s1.s:  # Move elements from s1 to s2
                self.s2.push(self.s1.pop())
        return self.s2.pop()
    
    def __str__(self) -> str:
        if self.s2.s:
            return str(self.s2)
        else:
            return str(self.s1)


if __name__ == '__main__':
    q = Queue()
    q.enqueue(1)
    q.enqueue(2)
    print(q.dequeue())
    q.enqueue(3)
    print(q.dequeue())
    print(q.dequeue())
    
