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
    
#missing number:
# can use the sum formula n(n+1)/2
# can use a set to track numbers present in the array
# can use the xor trick
# where we xor all numbers from 0 to n and all numbers in the array
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = len(nums)
        for i in range(len(nums)):
            res += (i- nums[i])
        return res
    


#find all numbers that disappeared in an array
# can use a set to track numbers present in the array
# then check which numbers from 1 to n are missing

class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        S = set()
        res = []
        for i in nums:
            if i not in S:
                S.add(i)
        for i in range(1, len(nums)+1):
            if i not in S:
                res.append(i)
        return res


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


#smaller numbers than current:
# can sort the array and use a dictionary to map each number to its index in the sorted array
# then iterate through the original array to get the count of smaller numbers
# can also use a counting sort approach if the range of numbers is known and small
# can use a brute force approach with nested loops, but it's O(n^2)

class Solution(object):
    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        sorted_list= sorted(nums)
        dic = {}
        for idx, val in enumerate(sorted_list):
            if val not in dic:
                dic[val] = idx

        res =[]
        for i in nums:
            res.append(dic[i])
        return res
    
#O(n) solution for smaller numbers than current:    
class Solution(object):
    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        count = [0] * 101  # Assuming numbers are in the range 0-100
        for num in nums:
            count[num] += 1
        
        for i in range(1, len(count)):
            count[i] += count[i - 1]
        
        return [count[num - 1]  for num in nums]

#minimum time to visit all points:
# can calculate the distance between each pair of points using the max of the absolute differences in x or y

class Solution(object):
    def minTimeToVisitAllPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        a,b = points.pop()
        res=0
        while points:
            c,d = points.pop()
            dist = max(abs(c-a), abs(d-b))
            res+=dist
            a,b = c,d
        return res

#spiral order matrix:
# can use a while loop to traverse the matrix in a spiral order
# keep track of the boundaries (left, right, top, bottom) and adjust them as you traverse
# use a list to collect the elements in spiral order.
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        res = []
        left = 0
        top = 0
        right = len(matrix[0])
        bottom = len(matrix)
        
        while left<right and top < bottom:
            for i in range(left, right):
                res.append(matrix[top][i])       
            top+=1

            for i in range(top, bottom):
                res.append(matrix[i][right-1])
            right-=1

            if not (left<right and top <bottom): break

            for i in range(right-1, left-1, -1):
                res.append(matrix[bottom-1][i])
            bottom -=1

            for i in range(bottom-1, top-1, -1):
                res.append(matrix[i][left])
            left+=1
        return res

#number of islands:
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        
        def dfs(i, j):
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
                return
            grid[i][j] = '0'  # Mark the land as visited
            dfs(i + 1, j)  # Down
            dfs(i - 1, j)  # Up
            dfs(i, j + 1)  # Right
            dfs(i, j - 1)  # Left
        
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':  # Found an island
                    count += 1
                    dfs(i, j)  # Visit all parts of the island
        
        return count
    

#Two pointers:
#best time to buy and sell stock:
# can use a two-pointer approach to find the maximum profit
# by keeping track of the minimum price seen so far and calculating the profit at each step 
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices or len(prices) < 2:
            return 0  # If there are fewer than 2 prices, there can be no profit
        l,r = 0,0
        maxProf = 0

        while r < len(prices):
            # profitable?
            if prices[l]< prices[r]:
                profit = prices[r] - prices[l] 
                maxProf = max(profit, maxProf)
            else:
                l = r
            r+=1
        return maxProf  
    

#sorted squares:
# can use a two-pointer approach to fill the result array with squares of the numbers in sorted order.  
class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = [0] * len(nums)  # preallocate result array
        l, r = 0, len(nums) - 1
        pos = len(nums) - 1  # fill result from end to start

        while l <= r:
            left, right = abs(nums[l]), abs(nums[r])
            if left > right:
                res[pos] = left * left
                l += 1
            else:
                res[pos] = right * right
                r -= 1
            pos -= 1
        return res
    
#3sum:
# can use a two-pointer approach after sorting the array    
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] > 0:
                break
            if i == 0 or nums[i - 1] != nums[i]:
                self.twoSumII(nums, i, res)
        return res

    def twoSumII(self, nums, i, res):
        lo, hi = i + 1, len(nums) - 1
        while (lo < hi):
            sum = nums[i] + nums[lo] + nums[hi]
            if sum < 0:
                lo += 1
            elif sum > 0:
                hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1

#longest mountain:
class Solution(object):
    def longestMountain(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        n = len(arr)
        if n < 3:
            return 0
        
        longest = 0
        i = 1
        
        while i < n - 1:
            # Check if arr[i] is a peak
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                left = i - 1
                right = i + 1
                
                # Expand to the left
                while left > 0 and arr[left] > arr[left - 1]:
                    left -= 1
                
                # Expand to the right
                while right < n - 1 and arr[right] > arr[right + 1]:
                    right += 1
                
                # Calculate the length of the mountain
                longest = max(longest, right - left + 1)
                
                # Move i to the end of the mountain
                i = right
            else:
                i += 1
        
        return longest
    
#conrains duplicate II:
#can use a set too.
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        hashmap = {}
        for i, n in enumerate(nums):
            if n in hashmap and i - hashmap[n] <= k:
                return True
            hashmap[n] = i
        return False

#minimum absolute difference:
class Solution(object):
    def minimumAbsDifference(self, arr):
        """
        :type arr: List[int]
        :rtype: List[List[int]]
        """
        arr.sort()
        min_diff = float('inf')
        result = []
        
        for i in range(1, len(arr)):
            diff = arr[i] - arr[i - 1]
            if diff < min_diff:
                min_diff = diff
                result = [[arr[i - 1], arr[i]]]
            elif diff == min_diff:
                result.append([arr[i - 1], arr[i]])
        
        return result

#minimum size subarray:
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        current_sum = 0
        min_length = float('inf')
        
        for right in range(len(nums)):
            current_sum += nums[right]
            
            while current_sum >= target:
                min_length = min(min_length, right - left + 1)
                current_sum -= nums[left]
                left += 1
        
        return min_length if min_length != float('inf') else 0
    
#single number:
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for n in nums:
            res ^= n
        return res

#coins change
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # Initialize dp array with amount + 1, which acts as "infinity"
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0  # Base case: 0 coins to make amount 0

        for a in range(1, amount + 1):
            for c in coins:
                if a - c >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - c])  # Choose the minimum coins needed

        # If dp[amount] has not been updated, return -1 (impossible case)
        return dp[amount] if dp[amount] != amount + 1 else -1
    

#climbing stairs:
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n
        
        first, second = 1, 2
        for i in range(3, n + 1):
            first, second = second, first + second
        
        return second
#mximum subarray:
# can use a sliding window approach to find the maximum subarray sum.
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_sub = nums[0]
        s = 0
        for i in nums:
            if s<0:
                s=0
            s +=  i
            max_sub = max(max_sub, s)
        return max_sub

#counting bits: 
class Solution(object):
    def countBits(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        dp = [0] * (n + 1)
        dp[0]=0
        offset = 1
        for i in range(1, n + 1):
            if offset * 2 == i:
                offset = i
            dp[i] = dp[i - offset] + 1
        return dp[n+1]

#range sum query immutable: 
class NumArray(object):
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums
        #adding an extra 0 at the start for easier prefix sum calculation.fixes edge cases.
        self.prefix_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.prefix_sum[i + 1] = self.prefix_sum[i] + nums[i]

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.prefix_sum[j + 1] - self.prefix_sum[i]

#middle of the linked list:
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
#linked list cycle:    
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False

#reverse linked list:
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        current = head
        
        while current:
            next_node = current.next  # Store the next node
            current.next = prev  # Reverse the link
            prev = current  # Move prev to current
            current = next_node  # Move to the next node
        
        return prev  # New head of the reversed list

#remove linked list elements:
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = ListNode(0)  # Create a dummy node to handle edge cases
        dummy.next = head
        current = dummy
        
        while current and current.next:
            if current.next.val == val:
                current.next = current.next.next  # Skip the node with the value
            else:
                current = current.next  # Move to the next node
        
        return dummy.next  # Return the new head of the list
    
#reverse linked list II:
class Solution(object):
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        if not head or left == right:
            return head
        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        # Move prev to the node before the left position
        for _ in range(left - 1):
            prev = prev.next
        
        # Start reversing from the left position
        current = prev.next
        next_node = None
        
        for _ in range(right - left + 1):
            next_node = current.next
            current.next = prev.next
            prev.next = current
            current = next_node
        
        # Connect the end of the reversed part to the rest of the list
        head.next = current
        
        return dummy.next
    
    #merge two sorted lists:
    # Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
 
        dummy = c = ListNode()
        while list1  and list2 :
            if list1.val < list2.val :
                c.next = list1
                list1 = list1.next 
            else:
                c.next = list2
                list2 = list2.next
            c = c.next
        if list1: c.next = list1
        if list2: c.next = list2
        
        return dummy.next
    
#min stack:
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        """
        :rtype: void
        """
        if self.stack:
            top = self.stack.pop()
            if top == self.min_stack[-1]:
                self.min_stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1] if self.stack else None

    def getMin(self):
        """
        :rtype: int
        """
        return self.min_stack[-1] if self.min_stack else None
    
#vlid parenthesis:
class Solution(object):
    
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for i in s:
            if i == "(" or i == "[" or i == "{":
                stack.append(i)
            elif stack and (i == ")" and stack[-1] == "(" or i == "]" and stack[-1] == "[" or i == "}" and stack[-1] == "{"):
                stack.pop()
            else:
                return False
        return not stack
    

#evakuate reverse polish notation:
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for token in tokens:
            if token in "+-*/":
                b = stack.pop()
                a = stack.pop()
                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)
                elif token == "/":
                    # Truncate toward zero
                    result = int(float(a) / b)
                    stack.append(result)
            else:
                stack.append(int(token))
        return int(stack[0])  # <- ensure integer return, not float
    
#stack sort:
class Solution(object):
    def sortStack(self, stack):
        """
        :type stack: List[int]
        :rtype: List[int]
        """
        temp_stack = []
        
        while stack:
            current = stack.pop()
            while temp_stack and temp_stack[-1] > current:
                stack.append(temp_stack.pop())
            temp_stack.append(current)
        
        return temp_stack[::-1]  # Return the sorted stack in ascending order
    
#stack using queues:
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = []

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.queue.append(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.queue.pop()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.queue[-1]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return len(self.queue) == 0
    
#time needed to buy tickets:
class Solution(object):
    def timeRequiredToBuy(self, tickets, k):
        """
        :type tickets: List[int]
        :type k: int
        :rtype: int
        """
        time = 0
        while tickets[k] > 0:
            for i in range(len(tickets)):
                if tickets[i] > 0:
                    tickets[i] -= 1
                    time += 1
                    if i == k and tickets[i] == 0:
                        return time
      
#average sum of levels in binary tree:
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root:
            return []
        
        from collections import deque
        queue = deque([root])
        averages = []
        
        while queue:
            level_sum = 0
            level_count = len(queue)
            
            for _ in range(level_count):
                node = queue.popleft()
                level_sum += node.val
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            averages.append(level_sum / level_count)
        
        return averages
# minimum depth of binary tree:

class Solution(object):
    def minDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if root is None:
            return 0

        # If one child is missing, we cannot take the min of 0 and non-zero — need to go down the non-null side.
        if root.left is None:
            return 1 + self.minDepth(root.right)
        if root.right is None:
            return 1 + self.minDepth(root.left)

        # If both children exist, take the min of both
        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))
    
#maximum depth of binary tree:
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if not root:
            return 0
        
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        
        return 1 + max(left_depth, right_depth)


#BT level order traversal:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res =[]
        q =collections.deque()    
        q.append(root)
        while q:
            qLen = len(q)
            level = []
            for i in range (qLen):
                node = q.popleft()
                if node:
                    level.append(node.val)
                    q.append(node.left)
                    q.append(node.right)
            if level:
                res.append(level)
        return res


#same tree:
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    
#path sum:
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: Optional[TreeNode]
        :type targetSum: int
        :rtype: bool
        """
        total= 0 
        if not root:
            return False
          # If it's a leaf, check if the value matches the remaining sum
        if not root.left and not root.right:
            return root.val == targetSum

        # Otherwise, subtract current node's value and recurse down
        remaining = targetSum - root.val
        return (
            self.hasPathSum(root.left, remaining) or
            self.hasPathSum(root.right, remaining))
    
#diameter of binary tree:
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.diameter = 0
        
        def depth(node):
            if not node:
                return 0
            
            left_depth = depth(node.left)
            right_depth = depth(node.right)
            
            # Update the diameter at this node
            self.diameter = max(self.diameter, left_depth + right_depth)
            
            # Return the depth of the tree rooted at this node
            return 1 + max(left_depth, right_depth)
        
        depth(root)
        return self.diameter
#invert binary tree:
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        root.right, root.left = root.left, root.right
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
        
#lowest common ancestor of a binary tree:
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        if not root:
            return None

        # If root is either p or q, we've found at least one node
        if root == p or root == q:
            return root

        # Recur on left and right subtrees
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        # If both sides returned non-null, this node is the LCA
        if left and right:
            return root

        # Otherwise return the non-null side (either left or right)
        return left if left else right

#seearch in a binary search tree:
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if not root:
            return None
        
        if root.val == val:
            return root
        elif val < root.val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)
    
#insesrt into a binary search tree:
class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if not root:
            return TreeNode(val)
        
        if val < root.val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)
        
        return root

#sorted array to bst:
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid + 1:])
        
        return root

#two sum IV - input is a BST:
class Solution(object):
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        if not root:
            return False

        seen = set()
        queue = deque([root])

        while queue:
            node = queue.popleft()

            if (k - node.val) in seen:
                return True
            seen.add(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return False
    
#lowest common ancestor of a binary search tree:
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        
        # If both p and q are smaller than root, LCA is in the left subtree
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        
        # If both p and q are greater than root, LCA is in the right subtree
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        
        # Otherwise, root is the LCA
        return root#
    
#minimum absolute difference in BST:    
class Solution(object):
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.prev = None
        self.min_diff = float('inf')
        
        def inorder(node):
            if not node:
                return
            
            inorder(node.left)
            
            if self.prev is not None:
                self.min_diff = min(self.min_diff, node.val - self.prev.val)
            self.prev = node
            
            inorder(node.right)
        
        inorder(root)
        return self.min_diff
    
#balance a binary search tree:
class Solution(object):
    def balanceBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        def inorder(node):
            if not node:
                return []
            return inorder(node.left) + [node.val] + inorder(node.right)
        
        def sortedArrayToBST(nums):
            if not nums:
                return None
            mid = len(nums) // 2
            root = TreeNode(nums[mid])
            root.left = sortedArrayToBST(nums[:mid])
            root.right = sortedArrayToBST(nums[mid + 1:])
            return root
        
        sorted_values = inorder(root)
        return sortedArrayToBST(sorted_values)

#delete node in a BST:
class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if not root:
            return None
        
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        else:
            # Node with one child or no child
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            
            # Node with two children: get the inorder successor (smallest in the right subtree)
            min_larger_node = self.getMin(root.right)
            root.val = min_larger_node.val
            root.right = self.deleteNode(root.right, min_larger_node.val)
        
        return root
    
    def getMin(self, node):
        while node.left:
            node = node.left
        return node
    
#top k frequent elements using heap without built-in functions:
import heapq
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if not nums or k <= 0:
            return []
        
        # Count frequency of each number
        frequency = {}
        for num in nums:
            frequency[num] = frequency.get(num, 0) + 1
        
        # Create a min-heap of size k
        min_heap = []
        
        for num, freq in frequency.items():
            if len(min_heap) < k:
                heapq.heappush(min_heap, (freq, num))
            else:
                heapq.heappushpop(min_heap, (freq, num))
        
        return [num for freq, num in min_heap]

# k closest points to origin using heap 
import heapq    
class Solution(object):
    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        if not points or k <= 0:
            return []
        
        # Create a max-heap based on the distance from the origin
        max_heap = []
        
        for x, y in points:
            dist = -(x**2 + y**2)  # Use negative distance for max-heap behavior
            if len(max_heap) < k:
                heapq.heappush(max_heap, (dist, [x, y]))
            else:
                heapq.heappushpop(max_heap, (dist, [x, y]))
        
        return [point for dist, point in max_heap]

#top k frequent elements using heap without built-in functions:
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if not nums or k <= 0:
            return []
        
        # Count frequency of each number
        frequency = {}
        for num in nums:
            frequency[num] = frequency.get(num, 0) + 1
        
        # Create a min-heap of size k
        min_heap = []
        
        for num, freq in frequency.items():
            if len(min_heap) < k:
                heapq.heappush(min_heap, (freq, num))
            else:
                heapq.heappushpop(min_heap, (freq, num))
        
        return [num for freq, num in min_heap]
    
#task scheduler without heap:
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        if not tasks:
            return 0
        
        # Count frequency of each task
        frequency = {}
        for task in tasks:
            frequency[task] = frequency.get(task, 0) + 1
        
        max_freq = max(frequency.values())
        max_count = sum(1 for freq in frequency.values() if freq == max_freq)
        
        # Calculate the minimum intervals needed
        intervals = (max_freq - 1) * (n + 1) + max_count
        
        return max(intervals, len(tasks))  # Ensure we don't return less than the number of tasks
    
#cheapest flights within k stops:   
class Solution(object):
    def findCheapestPrice(self, n, flights, src, dst, k):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type k: int
        :rtype: int
        """
        from collections import defaultdict
        import heapq
        
        graph = defaultdict(list)
        for u, v, w in flights:
            graph[u].append((v, w))
        
        # Min-heap to store (cost, current_node, stops)
        heap = [(0, src, 0)]
        min_cost = float('inf')
        
        while heap:
            cost, node, stops = heapq.heappop(heap)
            
            if node == dst:
                min_cost = min(min_cost, cost)
                continue
            
            if stops > k or cost > min_cost:
                continue
            
            for neighbor, price in graph[node]:
                heapq.heappush(heap, (cost + price, neighbor, stops + 1))
        
        return min_cost if min_cost != float('inf') else -1
    
#course schedule
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        premap = { i:[] for i in range(numCourses)}
        for crs, pre in prerequisites:
            premap[crs].append(pre)
        
        visit = set()
        def dfs(crs):
            if crs in visit:
                return False
            if premap[crs]==[]:
                return True
            
            visit.add(crs)
            for pre in premap[crs]:
                if not dfs(pre):
                    return False
            visit.remove(crs)
            premap[crs]= []
            return True

        for i in range(numCourses):
            if not dfs(i): return False
        return True