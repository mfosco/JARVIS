# input: n - a positive integer representing the grid size.
# output: number of valid paths from (0,0) to (n-1,n-1).
def numOfPathsToDest(n):
    # allocate a 2D array for memoization
    memo = [[-1 for x in range(n)] for y in range(n)]
    # the memoization array is initialized with -1
    # to indicate uncalculated squares.

    return numOfPathsToSquare(n-1, n-1, memo)

# input:
#    i, j - a pair of non-negative integer coordinates
#    memo - a 2D memoization array.
# output:
#    number of paths from (0,0) to the square represented in (i,j),
def numOfPathsToSquare(i, j, memo):
    if (i < 0 or j < 0):
        return 0
    elif (i < j):
        memo[i][j] = 0
    elif (memo[i][j] != -1):
        return memo[i][j]
    elif (i == 0 and j == 0):
        memo[i][j] = 1
    else:
        memo[i][j] = numOfPathsToSquare(i, j -1, memo) + \
            numOfPathsToSquare(i - 1, j, memo)
    return memo[i][j]

def get_indices_of_item_wights(arr, limit):
    indx = 0
    d = {}
    for a in arr:
        if a in d:
            return [indx, d[a]]
        d[limit - a] = indx
        indx += 1

    return []

def flight_path(points):
    maxy = max([x[2] for x in points])
    return maxy - points[0][2]