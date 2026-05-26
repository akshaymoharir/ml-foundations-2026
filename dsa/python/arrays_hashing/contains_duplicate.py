#!/usr/bin/env python3

from typing import List


def has_duplicate(nums: List[int]) -> bool:
    is_duplicate = False

    # Create a dictionary to hold seen values from input nums array
    # Max size of dictionary will be length of array if no duplicates
    length_of_nums = len(nums)
    print(f"Length of input array nums is: {length_of_nums}")
    seen_numbers = {}

    # Iterate over input array nums and add each element into dictionary if previously not seen
    for key, value in enumerate(nums):
        print(f"key:{key}", f"value:{value}")
        if value in seen_numbers.keys():
            print(f"This value is seen in array previously! Index:{seen_numbers[value]} and value:{value}")
            is_duplicate = True
            break
        else:
            seen_numbers[value] = key   # Store value from array as key and associate its id as value for faster processing

    return is_duplicate


if __name__ == "__main__":
    assert has_duplicate(nums=[67, 66, 65, 64, 63]) == False
    assert has_duplicate(nums=[2, 10, 15, 9, 20, 34, 97, 14, 78, 56, 9]) == True
