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