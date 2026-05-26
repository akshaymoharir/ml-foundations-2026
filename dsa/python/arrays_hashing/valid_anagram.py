#!/usr/bin/env python3

"""
Valid Anagram

Given two strings s and t, return true if the two strings are anagrams of each other, 
    otherwise return false.
An anagram is a string that contains the exact same characters as another string, 
    but the order of the characters can be different.

Example 1:
Input: s = "racecar", t = "carrace"
Output: true

Example 2:
Input: s = "jar", t = "jam"
Output: false

Constraints:
1 <= s.length, t.length <= 5 * 10^4
s and t consist of lowercase English letters.

"""

import logging

def parse_characters_from_string_into_dict(input_string: str) -> dict:
    output_dict = {}
    logging.info(f"input string to analyze:{input_string}")
    sorted_input_string = "".join(sorted(input_string))
    for (key, value) in enumerate(sorted_input_string ):
        if value in output_dict.keys():
            output_dict[value] += 1
        else:
            output_dict[value] = 1
    logging.info(f"Output dictionary prepared is:{output_dict}")
    return output_dict

def is_anagram(s:str, t:str) -> bool:

    is_anagram = True
    parsed_characters_dict_s = {}
    parsed_characters_dict_t = {}

    # input strings cant be anagram if their length is not same
    if not len(s) == len(t):
        is_anagram = False
        return is_anagram

    # Parse character from each string and store count for each character.
    # Use same function and parse and store count of each character.
    parsed_characters_dict_s = parse_characters_from_string_into_dict(s)
    parsed_characters_dict_t = parse_characters_from_string_into_dict(t)
    logging.info(f"parsed_string_s:{parsed_characters_dict_s}")
    logging.info(f"parseed_string_t:{parsed_characters_dict_t}")

    # if keys in dict are not equal strings cant be anagram
    if not len(parsed_characters_dict_s.keys()) == len(parsed_characters_dict_t.keys()):
        is_anagram = False
        return is_anagram

    check_second_loop = False
    # Compare for each key 2 dictionaries, values are same or not
    for key, value in parsed_characters_dict_s.items():
        if key in parsed_characters_dict_t.keys() and value == parsed_characters_dict_t[key]:
            # key value pair is same, continue scanning more key value pairs
            pass
        else:
            check_second_loop = True
    logging.info(f"anagram status for string {s} and {t} is: {is_anagram}")
    
    # Compare for each key 2 dictionaries, values are same or not
    if check_second_loop:
        for key, value in parsed_characters_dict_t.items():
            if key in parsed_characters_dict_s.keys() and value == parsed_characters_dict_s[key]:
                # key value pair is same, continue scanning more key value pairs
                pass
            else:
                is_anagram = False
                break
        logging.info(f"anagram status for string {s} and {t} is: {is_anagram}")

    return is_anagram

def test_anagram() -> None:
    test_cases = [("abcd", "dbac", True),
                  ("my_name_is_akshay", "mykshay___anamesi", True),
                  ("one_string", "another_string", False),
                  ("klmn", "pqrs", False)]
    
    for s, t, expected_result in test_cases:
        logging.info(f"\n\ntesting case:{s},{t}")
        actual_result = is_anagram(s=s, t=t)
        logging.info(f"Expected Result:{expected_result} \tActual Result:{actual_result}")
        assert expected_result == actual_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_anagram()
