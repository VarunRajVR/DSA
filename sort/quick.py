def quicky(arr):
    # Base case
    if len(arr) <= 1:
        return arr
    pivot = arr.pop()
    lower = []
    higher = []
    
    for i in arr:
        if i < pivot:
            lower.append(i)
        else:
            higher.append(i)
    return quicky(lower) + [pivot] + quicky(higher)

# def partition(arr, start, end):
#     pivot_ind = start
#     pivot = arr[start]
#     while start<end:
#         while start < len(arr) and arr[start]<=pivot:
#             start+=1
#         while arr[end]>pivot:
#             end-=1
#         arr[start], arr[end] = arr[end], arr[start]
#     arr[pivot_ind], arr[end] = arr[end], arr[pivot_ind]
#     return end

def swap(a, b, arr):
    if a!=b:
        tmp = arr[a]
        arr[a] = arr[b]
        arr[b] = tmp

def quick_sort(elements, start, end):
    if start < end:
        pi = partition(elements, start, end)
        quick_sort(elements, start, pi-1)
        quick_sort(elements, pi+1, end)

def partition(elements, start, end):
    pivot_index = start
    pivot = elements[pivot_index]

    while start < end:
        while start < len(elements) and elements[start] <= pivot:
            start+=1

        while elements[end] > pivot:
            end-=1

        if start < end:
            swap(start, end, elements)

    swap(pivot_index, end, elements)

    return end


if __name__ == '__main__':
    elements = [11,9,29,7,2,15,28]
    # elements = ["mona", "dhaval", "aamir", "tina", "chang"]
    quick_sort(elements, 0, len(elements)-1)
    print(elements)

    #Example usage
    arr = [3, 6, 8, 10, 1, 2, 1]
    sorted_arr = quicky(arr)
    print(sorted_arr)  

