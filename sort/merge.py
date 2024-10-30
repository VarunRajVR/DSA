def merge_arrays(a,b, arr):
    i=j=k=0
    while i<len(a) and j<len(b):
        if a[i]<b[j]:
            arr[k]= a[i]
            i+=1
        else:
            arr[k]= b[j]
            j+=1
        k+=1
    while i<len(a):
        arr[k]= a[i]
        i+=1
        k+=1
    while j<len(b):
        arr[k]= b[j]
        j+=1
        k+=1
def merge_sort(arr):
    if len(arr)<=1:
        return
    mid = len(arr)//2
    left = arr[mid:]
    right = arr[:mid]

    merge_sort(left)
    merge_sort(right)
    merge_arrays(left, right, arr)

# Example usage
lst = [12, 11, 13, 5, 6]
merge_sort(lst)
print(lst) 
    