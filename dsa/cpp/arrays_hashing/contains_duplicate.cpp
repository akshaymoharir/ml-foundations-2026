


/*

Neetcode: Contains Duplicate integer, (ported into C++)
Given an integer array nums, return true if any value appears more than once in the array, otherwise return false.

Example 1:
Input: nums = [1, 2, 3, 3]
Output: true

Example 2:
Input: nums = [1, 2, 3, 4]
Output: false

Constraints:
0 <= nums.length <= 10^5
-10^9 <= nums[i] <= 10^9

*/

#include <cassert>
#include <iostream>
#include <vector>
#include <unordered_map>

bool has_duplicate(std::vector<int16_t>& nums)
{
    bool has_duplicate = false;

    std::unordered_map<int16_t, bool> my_dict;

    for(auto& num: nums)
    {
        if (my_dict.contains(num))
        {
            has_duplicate = true;
            break;
        }
        else
        {
            my_dict[num] = true;
        }
    }

    return has_duplicate;
}

int main()
{
    std::vector<int16_t> test_1_nums = {1, 2, 3, 4, 46, 32, 11, 4, 6};
    std::vector<int16_t> test_2_nums = {0};
    std::vector<int16_t> test_3_nums = {0, 0};
    std::vector<int16_t> test_4_nums = {-51, 20, 37, -54, -46, 46, -11, 11, 4, 6, -20, 5, 20, 37, 5};
    std::vector<int16_t> test_5_nums = {101, 3452, 59, 34, 1, -567, -4562, 2, 3};
    std::vector<std::tuple<std::vector<int16_t>, bool>> tests = {
    std::make_tuple(test_1_nums, true),
    std::make_tuple(test_2_nums, false),
    std::make_tuple(test_3_nums, true),
    std::make_tuple(test_4_nums, true),
    std::make_tuple(test_5_nums, false),
    };
    
    int16_t test_num = 1;
    for (auto& [nums, expected] : tests)
    {
        bool result = has_duplicate(nums);
        bool expected_result = expected;
        if(result == expected_result)
        {
            std::cout << "Test:" << test_num << " PASSED\n";
        }
        else
        {
            std::cout << "Test " << test_num << " FAILED!! Expected result:" << expected_result << ", But got:" << result << "\n";
        }
        test_num++;
    }

    return 0;
}
