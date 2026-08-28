from collections import deque

def two_sum_sorted(nums, target):

    left = 0
    right = len(nums) - 1

    while left < right :

        current_sum = nums[left] + nums[right]

        if current_sum > target:
            right = right - 1
        elif current_sum < target:
            left = left + 1
        else:
            return [left, right]

    return None

def reverse_list(nums):

    left = 0
    right = len(nums) - 1

    while left < right:

        nums[left], nums[right] = nums[right], nums[left]

        left += 1
        right -= 1

    return nums

def maxWindowSum(nums, k):

    window_sum = sum(nums[0:k])

    for i in range(k, len(nums)-k+1):

        max = window_sum + nums[i] - nums[i-k]

        if max > window_sum:

           max = window_sum

    return max

def lengthOfLongestSubstring(s):

    left = 0
    seen = set()
    max_length = 0

    for right in range(len(s)):

        while s[right] in seen :

            seen.remove(s[left])
            left = left + 1

        seen.add(s[right])

        max_length = max(max_length, right - left + 1)

    return max_length

def minSubArrayLen(target, nums):

    left = 0
    current_sum = 0
    min_size = float("inf")

    for right in range(len(nums)):

        current_sum = current_sum + nums[right]

        while current_sum >= target:

            min_size = min(min_size, right - left + 1)

            current_sum = current_sum - nums[left]
            left = left + 1

    if min_size == float("inf"):

        min_size = 0

    return min_size

def isValid(s):

    mapping = {
        ")": "(",
        "]": "[",
        "}": "{"
    }

    stack = []

    for char in s:

        if char in mapping:

            # closing bracket
            if not stack:
                return False

            if stack[-1] != mapping[char]:
                return False

            stack.pop()

        else:

            # opening bracket
            stack.append(char)

    return len(stack) == 0

def processTasks(tasks):

    queue = deque()

    for i in range(len(tasks)):
        queue.append(tasks[i])

    while queue:
        task = queue.popleft()
        print(task)
    

    

