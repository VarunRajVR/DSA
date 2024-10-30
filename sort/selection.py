def selection(lst):
    size = len(lst)
    for i in range(size-1):
        small = i
        for j in range(i, size):
            if lst[small]>lst[j]:
                small = j
        if i!= small:
            lst[small], lst[i] = lst[i], lst[small]

lst = [2, 3, 1, 2, 5, 9, 8]
selection(lst)
print(lst) 

    
