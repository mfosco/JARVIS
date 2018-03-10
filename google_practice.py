def trap(height):
    if len(height) == 0:
        return 0
    ans = 0
    lim = len(height)
    left_max = [0]*lim
    right_max = [0]*lim
    left_max[0] = height[0]
    for i in range(1, lim):
        left_max[i] = max(height[i], left_max[i-1])
    right_max[lim-1] = height[lim-1]

    for i in range(lim-2, -1, -1):
        right_max[i] = max(height[i], right_max[i+1])

    for i in range(1, lim):
        ans += min(left_max[i], right_max[i]) - height[i]
    return ans, left_max, right_max

heights = [1,3,2,4,1,3,1,4,5,2,2,1,4,2,2]

ans, left, right = trap(heights)

import collections
import sys
def find_word_squares(words):
    # Preprocess words: O(#words * word-length) time and space
    words_by_letter_position = collections.defaultdict(set)
    for word in words:
        for index, letter in enumerate(word):
            words_by_letter_position[(index,letter)].add(word)
    # For each word, see if we can make a square.  O(#words * word-length^2/2)
    # for speed, assuming that set intersection is ~O(1) for small sets.
    # Worst-case storage is O(#words * word-length) for very very contrived
    # 'words'.  Normal English words will result in much smaller storage demand;
    # there is a practical maximum of ~170,000 English words.
    for word in words:
        # Initialize a set of incomplete possible squares; each item is an N-tuple
        # of words that are valid but incomplete word squares.  We could also do
        # depth-first via recursion/generators, but this approach is a little
        # simpler to read top-to-bottom.
        possible_squares = set([(word,)])
    # As long as we have any incomplete squares:
    while possible_squares:
        square = possible_squares.pop()
        # When matching an incomplete square with N words already present,
        # we need to match any prospective words to the tuples formed by
        # (N, Nth character in word 0), (N, Nth character in word 1), ...
        # Only words which satisfy all of those restrictions can be added.
        keys = [(i, square[i][len(square)]) for i in range(len(square))]
        possible_matches = [words_by_letter_position[key] for key in keys]
        for valid_word in set.intersection(*possible_matches):
            valid_square = square + (valid_word,)
            # Save valid square in 'ret' if it's complete, or save it as
            # a work-to-do item if it's not.
            if len(valid_square) == len(word):
                yield valid_square
            else:
                possible_squares.add(valid_square)

possibility = ['area', 'ball', 'dear', 'lady', 'lead', 'yard']

t = find_word_squares(possibility)

for item in t:
    print(item)


class Solution:
    # @return an integer
    def lengthOfLongestSubstring(self, s):
        start = maxLength = 0
        usedChar = {}

        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            else:
                maxLength = max(maxLength, i - start + 1)

            usedChar[s[i]] = i

        return maxLength

t = Solution()
s = "pwwkew"
print(t.lengthOfLongestSubstring(s))
