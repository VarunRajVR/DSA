class hash:
    def __init__(self) -> None:
        self.max = 10
        #init a list with None for max values
        self.d = [None for i in range(self.max)] # init as lists if you are implementing chaining. 

    
    def get_hash(self, key):
        return key % self.max
        # #handling string keys
        # for i in key:
        #     h += ord(i)
        # return h % self.max
    # allows using 
    def __setitem__(self, key, value):
        h = self.get_hash(key)

         # chaining
        # found = False
        # for index, element in enumerate(self.d[h]):
        #     if len(element)==2 and element[0]==key:
        #         self.d[h][index] = (key, value)
        #         found = True
        #         break
        # if not found:
        #     self.d[h].append((key,value))

        # linear probing
        for i in range(self.max):
            new_hash = (h + i) % self.max
            if self.d[new_hash] is None:
                self.d[new_hash] = (key, value)
                return
            elif self.d[new_hash][0] == key:
                self.d[new_hash] = (key, value)  # Update existing value
                return
        
        raise Exception("Hash table is full")  # Handling full table scenario
        
        

    def __getitem__(self, key):
        h = self.get_hash(key)
        for i in range(self.max):
            new_hash = (h + i) % self.max
            if self.d[new_hash] is None:
                continue
            if self.d[new_hash][0] == key:
                return self.d[new_hash][1]
        
        raise KeyError(f"Key {key} not found")

        
    
    def __delitem__(self, key):
        h= self.get_hash(key)

        for i in range(self.max):
            new_hash = (h + i) % self.max
            if self.d[new_hash] is None:
                continue
            if self.d[new_hash][0] == key:
                self.d[new_hash] = None
                return
        raise KeyError(f"Key {key} not found")

    def print(self):
        print(self.d)



if __name__ == '__main__':
    h = hash()
    h[6] = 43
    h[16] = 73
    print(h[6])
    h.print()  
    



    

        

        