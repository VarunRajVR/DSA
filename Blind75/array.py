# contains Duplicate:

# brute force n2
# can sort and use two pointers.
# can use hashset.
class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:
        S = set()
        for i in nums:
            if i not in S:
                S.add(i)
            else:
                return True
        return False
    
# Valid anagram.
#can use the get operator 

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False

       # Initialize dictionaries to count character occurrences
        S, T = {}, {}

        for char in s:
            if char in S:
                S[char] += 1
            else:
                S[char] = 1
        
        for char in t:
            if char in T:
                T[char] += 1
            else:
                T[char] = 1
        
        # Compare the two dictionaries
        return S == T
    

# two sum:
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for i,n in enumerate(nums):
            diff = target - n
            if diff in hashmap:
                return [ hashmap[diff],i ]
            hashmap[n] = i