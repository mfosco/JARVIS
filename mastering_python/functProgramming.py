# Ch. 4 Functional Programming
import pprint, functools, heapq, itertools
import random, operator, json, collections

squares         = [x ** 2 for x in range(10)]
uneven_squares  = [x ** 2 for x in range(10) if x % 2]


z = [random.random() for _ in range(10) if random.random() >= .5]
# numbers could be < .5 since first and last random calls are actually
# separate calls and return different results
print(z)

# a solution:
z = [x for _ in range(10) for x in [random.random()] if x >= .5]

# swapping column and row
matrix      = [
            [1, 2, 3, 4], 
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            ]

reshaped_matrix = [
        [
            [y for x in matrix for y in x] [i * len(matrix) + j]
            for j in range(len(matrix))
            ]
        for i in range(len(matrix[0]))
        ]

pprint.pprint(reshaped_matrix, width = 40)

# Dict comprehension
d_square    = {x: x ** 2 for x in range(10)}

# set comprehension
s_square    = {x*y for x in range(3) for y in range(3)}

# lambda functions (anonymous functions)
class Spam(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.value)

spams       = [Spam(5), Spam(2), Spam(4), Spam(1)]
sorted_spams= sorted(spams, key = lambda spam: spam.value)
print(spams)
print(sorted_spams)


Y = lambda f: lambda *args: f(Y(f)) (*args)
def factorial(combinator):
    def _factorial(n):
        if n:
            return(n * combinator(n-1))
        else:
            return(1)
    return(_factorial)

print(Y(factorial)(5))

# and short version
print(Y(lambda c: lambda n: n and n*c(n-1) or 1)(5))

# or ternary operator:
print(Y(lambda c: lambda n: n * c(n - 1) if n else 1)(5))

# quicksort definition
quicksort   = Y(lambda f:
        lambda x: (
            f([item for item in x if item < x[0]])
            + [y for y in x if x[0] == y]
            + f([item for item in x if item > x[0]])
            ) if x else [])

print(quicksort([1, 3, 5, 4, 1, 3, 2]))

heap        = []
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 4)
print(heapq.nsmallest(3, heap))

# alternative
heap        = []
push        = functools.partial(heapq.heappush, heap)
smallest    = functools.partial(heapq.nsmallest, iterable = heap)
push(1)
push(3)
push(5)
push(2)
push(4)
smallest(3)

# reduce
print(functools.reduce(operator.mul, range(1, 6)))

def tree():
    return(collections.defaultdict(tree))

# build tree
taxonomy    = tree()
reptilia    = taxonomy['Chordata']['Vertebrata']['Reptilia']
reptilia['Squamata']['Serpentes']['Pythonidas'] = ['Liasis', 'Morelia',
        'Python']

print(json.dumps(taxonomy, indent=4))

# path we want
path        = 'Chordata.Vertebrata.Reptilia.Squamata.Serpentes'

pathy       = path.split('.')
family      = functools.reduce(lambda a, b: a[b], pathy, taxonomy)
print(family.items())

suborder    = functools.reduce(lambda a, b: a[b], pathy, taxonomy)
print(suborder.keys())


months      = [10, 8, 5, 7, 12, 10, 5, 8, 15, 3, 4, 2]
print(list(itertools.accumulate(months, operator.add)))
print(list(itertools.accumulate(months))) # add is default

a           = range(3)
b           = range(5)
print(itertools.chain(a, b))
# if you have an iterable containing iterables, want to use
# itertools.chain.from_iterable

# combinations
print(list(itertools.combinations(range(3), 2)))

# with replacement
print(list(itertools.combinations_with_replacement(range(3), 2)))

# get that powerset
def powerset(iterable):
    return(itertools.chain.from_iterable(
        itertools.combinations(iterable, i)
        for i in range(len(iterable) + 1)))

print(list(powerset(range(3))))

# permutations
print(list(itertools.permutations(range(3), 2)))

# compress applies a boolean filter to your iterable
print(list(itertools.compress(range(1000), [0, 1, 1, 1, 0, 1])))

# dropwhile and takewhile
print(list(itertools.dropwhile(lambda x: x <= 3, [1, 3, 5, 4, 2])))

print(list(itertools.takewhile(lambda x: x <= 3, [1, 3, 5, 4, 2])))

# COUNT IS INFINITE SO DON'T DO: list(itertools.count()) but you can use floats

for a, b in zip(range(3), itertools.count()):
    print(a, b)

for a, b in zip(range(5, 10), itertools.count(5, .5)):
    print(a, b)

# groupby
# note: input needs to be sorted by the group parameter. Else it will be added
# as a separate group. ALSO, results are available for use only once
items       = [('a', 1), ('a', 2), ('b', 2), ('b', 0), ('c', 3)]
for group, items in itertools.groupby(items, lambda x: x[0]):
    print('%s: %s' % (group, [v for k , v in items]))

items       = [('a', 1), ('a', 2), ('b', 2), ('b', 0), ('c', 3)]
# unexpected results:
groups      = dict()
for group, items in itertools.groupby(items, lambda x: x[0]):
    groups[group]   = items
    print('%s: %s' % (group, [v for k, v in items]))

for group, items in sorted(groups.items()):
    print('%s: %s' % (group, [v for k, v in items]))

# islice, slicing any iterable
z           = list(itertools.islice(itertools.count(), 2, 7))
print(z)


