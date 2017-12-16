import functools

def spam(key, value, list_=[], dict_={}):
    list_.append(value)
    dict_[key] = value

    print('List: %r' % list_)
    print('Dict: %r' % dict_)

spam('key 1', 'value 1')
spam('key 2', 'value 2')

# get unexpected output since default of list_ and dict_ are shared between
# multiple calls

# safer alternative
def spam(key, value, list_=None, dict_ = None):
    if list_ is None:
        list_ = []
    if dict_ is None:
        dict_ = {}

    list_.append(value)
    dict_[key] = value


# note, class properties will change in multiple locations if they are inherited

class Spam(object):
    list_ = []
    dict_ = {} #these two will be shared across all instances

    def __init__(self, thing)
        self.thing = thing # limited to a single instance

'''
dict_ = {'spam': 'eggs'}

can't do
for key in dict_:
    del dict_[key]

can do
for key in list(dict_):
    del dict_[key]
'''

def spamy(value):
    exception = None
    try:
        value = int(value)
    except ValueError as e:
        exception = e
        print('We caught an exception: %r' % exception)
    return(exception)

eggy = [lambda a: i * a for i in range(3)]

for egg in eggy:
    print(egg(5))

'''
above actually prints 10 10 10 due to late binding
'''

# a solution to above
eggy = [functools.partial(lambda i, a: i*a, i) for i in range(3)]
for egg in eggy:
    print(egg(5))


