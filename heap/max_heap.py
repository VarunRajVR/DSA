class Heap:
    def __init__(self):
        self.h = []

    def insert(self, ele):
        #insert new element at the end of the list
        self.h.append(ele)
        n = len(self.h)
        i = n - 1
        while i > 0:
            #take parent as the new element and keep checking the parent. 
            parent = (i - 1) // 2
            if self.h[i] > self.h[parent]:
                self.h[i], self.h[parent] = self.h[parent], self.h[i]
                i = parent
            else:
                break
    
    def remove(self):
        if len(self.h) == 0:
            print("Heap is empty")
            return None
        if len(self.h) == 1:
            return self.h.pop()

        # Store the root value.
        root_value = self.h[0]
        # Replace the first element with the first one. 
        self.h[0] = self.h.pop()

        # Heapify down from the root
        self.heapify_down(0)
        return root_value

    def heapify_down(self, index):
        n = len(self.h)
        largest = index
        left = 2 * index + 1
        right = 2 * index + 2
        #check if there is a left child and which is larger
        if left < n and self.h[left] > self.h[largest]:
            largest = left
        #check if there is a right child and which is larger
        if right < n and self.h[right] > self.h[largest]:
            largest = right
        # reset the largest value if needed
        if largest != index:
            self.h[index], self.h[largest] = self.h[largest], self.h[index]
            
            self.heapify_down(largest)

    def print(self):
        print(self.h)

if __name__ == '__main__':
    h = Heap()
    h.insert(5)
    h.insert(10)
    h.insert(2)
    h.insert(20)
    h.print()  
    print("Removed:", h.remove())  
    h.print()  
    print("Removed:", h.remove())  
    h.print() 
    print("Removed:", h.remove()) 
    h.print()  
    print("Removed:", h.remove())  
    h.print() 
    print("Removed:", h.remove()) 
