class LRU_cache():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.hash = {}  # Stores the key and the index of the key in the queue
        self.q = []  # Acts as the queue to maintain order of usage

    def put(self, key):
        value = key
        if key not in self.hash:
            if len(self.q) == self.capacity:
                # Cache is full, remove the least recently used item
                lru_key = self.q.pop(0)  # Remove the first item from the queue
                del self.hash[lru_key]  # Remove it from the hash map
            
            # Add the new key to the queue and hash map
            self.q.append(value)
            self.hash[key] = len(self.q) - 1  # Update the hash with the new index
        else:
            # If key is already in cache, move it to the end to mark it as recently used
            self.q.remove(key)
            self.q.append(value)
            self.hash[key] = len(self.q) - 1  # Update the index in the hash map
    
    def get(self, key):
        if key in self.hash:
            # Move the accessed key to the end of the queue to mark it as recently used
            self.q.remove(key)
            self.q.append(key)
            self.hash[key] = len(self.q) - 1  # Update the index in the hash map
            return key
        else:
            return "Cache fault!"

    def print(self):
        print(self.q)
        
if __name__ == '__main__':
    cache = LRU_cache(3)  
    print(cache.get(2))  # Cache fault
    cache.put(2)
    cache.print()  
    cache.put(3)
    cache.put(4)
    cache.print() 
    print(cache.get(2)) 
    cache.print()  # [3, 4, 2]
    cache.put(5) 
    cache.print()  # [4, 2, 5]
