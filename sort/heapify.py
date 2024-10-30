def heapify_down(lst, index, n):
    largest = index
    left = 2 * index + 1
    right = 2 * index + 2

    # Check if the left child exists and is larger than the current largest
    if left < n and lst[left] > lst[largest]:
        largest = left

    # Check if the right child exists and is larger than the current largest
    if right < n and lst[right] > lst[largest]:
        largest = right

    # Swap and continue heapifying if the largest is not the root
    if largest != index:
        lst[index], lst[largest] = lst[largest], lst[index]
        heapify_down(lst, largest, n)

def heap_sort(lst):
    n = len(lst)
    
    # Build a max-heap
    for i in range(n // 2 - 1, -1, -1):
        heapify_down(lst, i, n)
    
    # Extract elements from the heap
    for i in range(n-1, 0, -1):
        lst[i], lst[0] = lst[0], lst[i]  # Swap
        heapify_down(lst, 0, i)  # Heapify the root

# Example usage
lst = [2, 3, 1, 2, 5, 9, 8]
heap_sort(lst)
print(lst)  # The list should be sorted in ascending order
