def bubble_sort(lst):
    n = len(lst)
    for i in range(n):
        swapped = False
        # Last i elements are sorted
        for j in range(0, n-i-1): 
            # Swap if the element found is greater than the next element
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
                swapped = True
        if not swapped:
            break

# Example usage
lst = [2, 3, 1, 2, 5, 9, 8]
bubble_sort(lst)
print(lst) 
