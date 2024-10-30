def insertion_sort(lst):
    # Traverse through 1 to len(lst)
    for i in range(1, len(lst)):
        key = lst[i]
        j = i - 1

        # Move elements of list that are greater than key to one position ahead of their current position.
        while j >= 0 and key < lst[j]:
            lst[j + 1] = lst[j]
            j -= 1
        
        # Place the key in its correct position
        lst[j + 1] = key

# Example usage
lst = [12, 11, 13, 5, 6]
insertion_sort(lst)
print(lst) 
