
"""
Given an array of strings strs, group all anagrams together into sublists. You may return the output in any order.
An anagram is a string that contains the exact same characters as another string, but the order of the characters can be different.

Example 1:
Input: strs = ["act","pots","tops","cat","stop","hat"]
Output: [["hat"],["act", "cat"],["stop", "pots", "tops"]]

Example 2:
Input: strs = ["x"]
Output: [["x"]]

Example 3:
Input: strs = [""]
Output: [[""]]

Constraints:
1 <= strs.length <= 1000.
0 <= strs[i].length <= 100
strs[i] is made up of lowercase English letters.

"""

import logging
from dataclasses import dataclass

@dataclass
class InputStringAnalysis:
    id: int
    word_analysis = dict


def group_anagram(strs: list[str]) -> list[list[str]]:
    pass

def test_group_anagram()->None:
    test_cases = [("abcd", "dbac", True),
                  ("my_name_is_akshay", "mykshay___anamesi", True),
                  ("one_string", "another_string", False),
                  ("klmn", "pqrs", False)]
    
    for s, t, expected_result in test_cases:
        logging.info(f"\n\ntesting case:{s},{t}")
        actual_result = group_anagram(s=s, t=t)
        logging.info(f"Expected Result:{expected_result} \tActual Result:{actual_result}")
        assert expected_result == actual_result
    pass

if __name__ == "__main__":
    test_group_anagram()