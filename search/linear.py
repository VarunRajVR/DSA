def linear_search(lst, key):    
    found = False
    for i in range(len(lst)):
        if lst[i] == key:  
            print(f"Key: {key} found at index {i}")
            found = True
            break  

    if not found:
        print(f"Key: {key} not found")


lst = [2,7,4,6,2,1,2]
linear_search(lst, 1)
linear_search(lst, 18)
