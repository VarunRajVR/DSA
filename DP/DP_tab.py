import sys
sys.setrecursionlimit(10000)  # Increase the recursion limit if necessary



# fibonacci Series 

def fib(n):
    table = [0]*(n+1)
    table[1]=1
    for i in range(n):
        if i + 1 <= n:
            table[i + 1] += table[i]
        if i + 2 <= n:
            table[i + 2] += table[i]
    return table[n]

# grid traverse problem:
# start at top-left, Goal: bottom-right, allowed only down or right operations,
#  figure out how many ways to reach goal. 

def go_grid(m,n):
    table = [[0 for _ in range(n+1)] for _ in range(m+1)]
    table[1][1] = 1
    for i in range(m+1):
        for j in range(n+1):
            curr = table[i][j]
            if j+1 <=n : table[i][j+1] +=curr
            if i+1 <=m :table[i+1][j] +=curr
    return table[m][n]




#target Sum:
#Given a target and array, find if we can get the target from the array. #numbers can be repeatedly used
def canSum(target, nums):
    table = [False] * (target+1)
    table[0] = True
    for i in range(len(table)):
        if(table[i]==True):
            for num in nums:
                if i + num <= target: table[i+num]= True
    return table[target]


#howSum:
#Given a target and array, find if we can get the target from the array and return atleast one combination of the same.
#numbers can be repeatedly used
def howSum(target, nums):
    table = [None] * (target+1)
    table[0] = []
    for i in range(len(table)):
        if(table[i]is not None):
            for num in nums:
                if i + num <= target:
                    table[i+num]= table[i]
                    table[i]+= [num]
    return table[target]
    

#best sum:
# #Given a target and array, find if we can get the target from the array and return shortest combination of the same.

def bestSum(target, nums, memo = None):
    
    table = [None] * (target+1)
    table[0] = []
    for i in range(len(table)):
        if(table[i]is not None):
            for num in nums:
                if i + num <= target:
                    combination = table[i] +[num]
                    if not table[i+num] or len(table[i+num]) > len(combination):
                        table[i+num] = combination

                    
    return table[target]
    
            



# print(howSum(3000, [30000,1000]))
print(bestSum(8, [2,3,5]))
# print(canSum(3000, [1,4]))
# print(go_grid(3,3))
# print(fib(50))

