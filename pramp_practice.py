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
    return max(maxy - points[0][2],0)


# A node
class Node:
    # Constructor to create a new node
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


# A binary search tree
class BinarySearchTree:
    # Constructor to create a new BST
    def __init__(self):
        self.root = None

    def next_legit_right(self, inputNode):
        while inputNode.left != None:
            inputNode = inputNode.left
        return inputNode

    def find_in_order_successor(self, inputNode):
        if inputNode.right != None:
            # I am a parent so get next min key to the right
            return self.next_legit_right(inputNode.right)
        ancestor    = inputNode.parent
        child       = inputNode
        while ancestor != None and child == ancestor.right:
            child       = ancestor
            ancestor    = child.parent
        return ancestor

    # Given a binary search tree and a number, inserts a
    # new node with the given number in the correct place
    # in the tree. Returns the new root pointer which the
    # caller should then use(the standard trick to avoid
    # using reference parameters)
    def insert(self, key):

        # 1) If tree is empty, create the root
        if (self.root is None):
            self.root = Node(key)
            return

        # 2) Otherwise, create a node with the key
        #    and traverse down the tree to find where to
        #    to insert the new node
        currentNode = self.root
        newNode = Node(key)
        while (currentNode is not None):

            if (key < currentNode.key):
                if (currentNode.left is None):
                    currentNode.left = newNode;
                    newNode.parent = currentNode;
                    break
                else:
                    currentNode = currentNode.left;
            else:
                if (currentNode.right is None):
                    currentNode.right = newNode;
                    newNode.parent = currentNode;
                    break
                else:
                    currentNode = currentNode.right;

    # Return a reference to a node in the BST by its key.
    # Use this method when you need a node to test your
    # findInOrderSuccessor function on
    def getNodeByKey(self, key):

        currentNode = self.root
        while (currentNode is not None):
            if (key == currentNode.key):
                return currentNode

            if (key < currentNode.key):
                currentNode = currentNode.left
            else:
                currentNode = currentNode.right

        return None


#########################################
# Driver program to test above function #
#########################################

# Create a Binary Search Tree
bst = BinarySearchTree()
bst.insert(20)
bst.insert(9);
bst.insert(25);
bst.insert(5);
bst.insert(12);
bst.insert(11);
bst.insert(14);

# Get a reference to the node whose key is 9
test = bst.getNodeByKey(11)

# Find the in order successor of test
succ = bst.find_in_order_successor(test)

# Print the key of the successor node
if succ is not None:
    print("\nInorder Successor of %d is %d " \
          % (test.key, succ.key))
else:
    print("\nInorder Successor doesn't exist")


def get_cheapest_cost(rootNode):
    cost = rootNode.cost
    children = rootNode.children
    if len(children) == 0:
        return cost

    minChild = 100005

    for c in children:
        tmpCost = get_cheapest_cost(c)
        if tmpCost < minChild:
            minChild = tmpCost

    return cost + minChild


##########################################
# Use the helper code below to implement #
# and test your function above           #
##########################################

# A node
class Node:
    # Constructor to create a new node
    def __init__(self, cost):
        self.cost = cost
        self.children = []
        self.parent = None

    def addChildren(self, children):
        self.children += children
        for child in children:
            child.parent = self


rootNode = Node(0)

node1 = Node(5)
node2 = Node(3)
node3 = Node(6)
rootNode.addChildren([node1, node2, node3])

node4 = Node(4)
node1.addChildren([node4])

node5 = Node(2)
node6 = Node(0)
node2.addChildren([node5, node6])

node7 = Node(1)
node8 = Node(5)
node3.addChildren([node7, node8])

node9 = Node(1)
node5.addChildren([node9])

node10 = Node(10)
node6.addChildren([node10])

node11 = Node(1)
node9.addChildren([node11])

print(get_cheapest_cost(rootNode))


def spiral_copy(inputMatrix):
    n = len(inputMatrix)
    m = len(inputMatrix[0])
    res = []
    top_wall = 0
    bottom_wall = n - 1
    left_wall = 0
    right_wall = m - 1

    while top_wall <= bottom_wall and left_wall <= right_wall:

        for i in range(left_wall, right_wall + 1):
            res.append(inputMatrix[top_wall][i])
        top_wall += 1

        for j in range(top_wall, bottom_wall + 1):
            res.append(inputMatrix[j][right_wall])
        right_wall -= 1

        row_loc = bottom_wall
        if top_wall <= bottom_wall:
            for i in range(right_wall, left_wall - 1, -1):
                res.append(inputMatrix[bottom_wall][i])

            bottom_wall -= 1

        if left_wall <= right_wall:
            for j in range(bottom_wall, top_wall - 1, -1):
                res.append(inputMatrix[j][left_wall])
            left_wall += 1

    return res
