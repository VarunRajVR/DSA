def binary_search(lst, key):
    left, right = 0, len(lst) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if lst[mid] == key:
            return f"Key: {key} found at index {mid}"
        
        elif lst[mid] > key:
            right = mid - 1

        else:
            left = mid + 1

    return f"Key: {key} not found"


sorted_list = [1, 2, 4, 6, 7, 9, 12]
print(binary_search(sorted_list, 7))  
print(binary_search(sorted_list, 5))  
