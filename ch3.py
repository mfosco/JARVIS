import builtins, enum
import collections, json
import argparse, math
import bisect

primes          = set((1, 2, 3, 5, 7))
items           = list(range(10))

# classic sol
for prime in primes:
    items.remove(prime)
print(items)

# these latter 2 methods are much faster
# list comprehension
items           = list(range(10))
z               = [item for item in items if item not in primes]
print(z)

# filter
items           = list(range(10))
z               = list(filter(lambda item: item not in primes, items))
print(z)


# a poor hash
def most_significant(value):
    while value >= 10:
        value //= 10
    return(value)

def eggies(*args):
    print('args:', args)

# Collections
# mappings        = collections.ChainMap(globals(), locals(), vars(builtins))
# value           = mappings[key]

defaults        = {
                    'spam': 'default spam value',
                    'eggs': 'default eggs value', 
                }
parser          = argparse.ArgumentParser()
parser.add_argument('--spam')
parser.add_argument('--eggs')

args            = vars(parser.parse_args())
filtered_args   = {k: v for k, v in args.items() if v}
combined        = collections.ChainMap(filtered_args, defaults)
print(combined['spam'])

counter         = collections.Counter()
for i in range(0, 100000):
    counter[math.sqrt(i) // 25] += 1

for key, count in counter.most_common(5):
    print('%s: %d' %(key, count))

def tree():
    return(collections.defaultdict(tree))

colours         = tree()
colours['other']['black']   = 0x000000
colours['other']['white']   = 0xFFFFFF
colours['primary']['red']   = 0xFF0000
colours['primary']['green'] = 0x00FF00
colours['primary']['blue']  = 0x0000FF
colours['secondary']['yellow']  = 0xFFFF00
colours['secondary']['aqua']    = 0x00FFFF
colours['secondary']['fuchsia'] = 0xFF00FF
print(json.dumps(colours, sort_keys = True, indent = 4))


Point       = collections.namedtuple('Point', ['x', 'y', 'z'])
point_a     = Point(1, 2, 3)
print(point_a)


class Color(enum.Enum):
    red     = 1
    green   = 2
    blue    = 3

print(Color.red)
print(Color['red'])
print(Color(1))
print(Color.red.name)
print(Color.red.value)
print(isinstance(Color.red, Color))
print(Color.red is Color['red'])

for color in Color:
    print(color)


class Spamy(str, enum.Enum):
    EGGS        = 'eggs'

print(Spamy.EGGS == 'eggs')

# Ordered dict keeps track of when keys were inserted
# though keys can be ordered later
spammier    = collections.OrderedDict()
spammier['b']   = 2
spammier['c']   = 3
spammier['a']   = 1

print(spammier)

eglet   = collections.OrderedDict(sorted(spammier.items()))
print(eglet)

# bisect module keeps your list always sorted
sorted_list     = []
bisect.insort(sorted_list, 5)
bisect.insort(sorted_list, 3)
bisect.insort(sorted_list, 1)
bisect.insort(sorted_list, 2)
print(sorted_list)

listy       = [1, 2, 3, 5]
def contains(sorted_list, value):
    i   = bisect.bisect_left(sorted_list, value)
    return(i < len(sorted_list) and sorted_list[i] == value)

print(contains(listy, 2))
print(contains(listy, 4))
print(contains(listy, 6))
