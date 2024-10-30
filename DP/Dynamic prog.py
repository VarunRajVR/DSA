import sys
sys.setrecursionlimit(10000)  # Increase the recursion limit if necessary



# fibonacci Series 

def fib(n, memo = {}):
    #use hashmap for memoization to get O(n), if list, we get n2.
    if n in memo: return memo[n]
    #base case
    if n <=2: return 1
    #recursive case
    memo[n] = fib(n-1) + fib (n-2)
    #return value
    return memo[n]

# grid traverse problem:
# start at top-left, Goal: bottom-right, allowed only down or right operations,
#  figure out how many ways to reach goal. 

def go_grid(m,n, memo={}):
    #ddefine the key for uniformity
    key = str(m) + ',' +  str(n)
    # base case
    if key in memo: return memo[key]
    if m == 1 and n == 1: return 1
    if m==0 or n == 0: return 0
    #recursive call
    memo[key] = go_grid(m-1, n, memo) + go_grid(m, n-1,memo)
    return memo[key]



#target Sum:
#Given a target and array, find if we can get the target from the array. #numbers can be repeatedly used
def canSum(target, nums, memo ={}):
    #base cases
    if target == 0: return True
    if target <0 : return False
    if target in memo: return memo[target]

    for i in nums:
        rem = target -i
        if canSum(rem, nums,memo) == True:
            memo[target]=True
            return True
    
    memo[target]= False
    return False


#howSum:
#Given a target and array, find if we can get the target from the array and return atleast one combination of the same.
#numbers can be repeatedly used
def howSum(target, nums, memo = None):
    if memo is None: memo = {}
    if target == 0: return []
    if target <0 : return None
    if target in memo: return memo[target]

    for i in nums:
        rem = target -i
        result = howSum(rem, nums,memo)
        if result != None:
            memo[target]=result + [i]
            return memo[target]
    
    memo[target]= None
    return None

#best sum:
# #Given a target and array, find if we can get the target from the array and return shortest combination of the same.

def bestSum(target, nums, memo = None):
    if memo is None: memo = {}
    if target == 0: return []
    if target <0 : return None
    if target in memo: return memo[target]
    shortest_combi = None

    for i in nums:
        rem = target -i 
        result = bestSum(rem, nums,memo)
        if result != None:
            combination =result + [i]
            if shortest_combi == None or len(shortest_combi)> len(combination):
                shortest_combi = combination
    
    memo[target] = shortest_combi
    return memo[target]
            
#can Construct:
# given a target and set of strings, return if we can form the target from the set given. #strings can be reused. 

def canConstruct(target, words, memo ={}):
    if target == '': return True
    if target in memo: return memo[target]


    for i in words:
        if target.startswith(i):
            suffix = target[len(i):]
            if canConstruct(suffix, words,memo):
                memo[target] = True
                return memo[target]
    memo[target] = False
    return memo[target]

# Count Construct:
# given a target and set of strings, return number of ways we can form the target from the set given. #strings can be reused.

def countConstruct(target, words, memo ={}):
    if target == '': return 1
    if target in memo: return memo[target]
    total = 0
    for i in words:
        if target.startswith(i):
            suffix = target[len(i):]
            numwaysforrest = countConstruct(suffix, words,memo)
            total+= numwaysforrest
    memo[target] = total
    return total


# all Construct:
# given a target and set of strings, return all combinationse that form the target from the set given. #strings can be reused.
def allConstruct(target, words, memo ={}):
    if target == '': return [[]]

    if target in memo: return memo[target]
    total = []

    for i in words:
        if target.startswith(i):
            suffix = target[len(i):]
            combination = allConstruct(suffix, words,memo)
            targetways = [[i]+combi for combi in combination ]
            total += targetways

    memo[target] = total
    return total



# print(howSum(3000, [30000,1000]))
# print(bestSum(3000, [300,100]))
# print(canSum(3000, [1,4]))
# print(go_grid(5,5))
# print(fib(50))

# print(canConstruct('abcd', ['ab', 'ef', 'cd']))
# print(countConstruct('abcd', ['ab', 'ef', 'cd', 'abcd']))
print(allConstruct('abcd', ['ab', 'ef', 'cd', 'abcd']))