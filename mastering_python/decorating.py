# Ch 5
# Decorators

import functools, pprint

# print all arguments sent to spam and pass them to spam unmodified
def eggs(function):
    @ functools.wraps(function)
    def _eggs(*args, **kwargs):
        print('%r got args: %r and kwargs %r' % (function.__name__, args,
            kwargs))
        return(function(*args, **kwargs))

    return(_eggs)

@eggs
def spam(a, b, c):
    '''The spam function Returns a * b + c'''
    return(a * b + c)


# works but messy to do
def spammy(eggs):
    output      = 'spam' * (eggs % 5)
    print('spam(%r): %r' % (eggs, output))
    return(output)

# decorator for debugging
def debug(function):
    @functools.wraps(function)
    def _debug(*args, **kwargs):
        output      = function(*args, **kwargs)
        print('%s(%r, %r): %r' % (function.__name__, args, kwargs, output))
        return(output)
    return(_debug)

@debug
def spammish(eggs):
    return('spam' * (eggs % 5))

# memoization example
def memoize(function):
    function.cache      = dict()

    @functools.wraps(function)
    def _memoize(*args):
        if args not in function.cache:
            function.cache[args]        = function(*args)
        return(function.cache[args])
    return(_memoize)

@memoize
def fibonacci(n):
    if n < 2: 
        return(n)
    else:
        return(fibonacci(n-1) + fibonacci(n-2))

for i in range(1, 7):
    print(' fibonacci %d: %d' % (i, fibonacci(i)))

# much more inefficient method of calculating fibonacci
def fib(n):
    if n < 2:
        return(n)
    else:
        return(fib(n-1) + fib(n-2))

# see how that lru_cache goes

# Make a simple call counting decorator
def counter(function):
    function.calls      = 0
    @functools.wraps(function)
    def _counter(*args, **kwargs):
        function.calls  += 1
        return(function(*args, **kwargs))
    return(_counter)

# make a LRU cache with size 3
@functools.lru_cache(maxsize = 3)
@counter
def fibby(n):
    if n < 2:
        return(n)
    else:
        return(fibby(n - 1) + fibby(n - 2))

print(fibby(100))
print(fibby.__wrapped__.__wrapped__.calls)

# decorators with optional args

def add(*args, **kwargs):
    '''Add n to the input of the decorated function'''
    # The default kwargs, we don't want to store this in kwargs
    # because we want to make sure that args and kwargs
    # can't both be filled
    default_kwargs  = dict(n = 1)

    # The inner function, note that this is the actual decorator
    def _add(function):
        # the actual function that will be called
        @functools.wraps(function)
        def __add(n):
            default_kwargs.update(kwargs)
            return(function(n + default_kwargs['n']))
        return(__add)

    if len(args) == 1 and callable(args[0]) and not kwargs: 
        # Decorator call w/o arguments, just call it ourselves
        return(_add(args[0]))
    elif not args and kwargs:
        # Decorator call with arguments, this time it will
        # automatically be executed with function as the
        # first argument
        default_kwargs.update(kwargs)
        return(_add)
    else:
        raise(RuntimeError('This decorator only supports keyword arguments'))

@add
def alpha(n):
    return('eggs' * n)

@add(n=3)
def beta(n):
    return('beta' * n)

#@add(3) # will force runtime error
#def bacon(n):
#    return('bacon' * n)

# now to make decorators using classes
class Debug(object):
    def __init__(self, function):
        self.function       = function
        # functools.wraps for classes
        functools.update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        output      = self.function(*args, **kwargs)
        print('%s(%r, %r): %r' % (self.function.__name__, args, kwargs, output))
        return(output)

# only real diff is that functools.wraps is replaced with
# functools.update_wrapper in the __init__ method
@Debug
def sp(eggs):
    return('spam' * (eggs % 5))

print(sp(3))

# decorating class functions

def plus_one(function):
    @functools.wraps(function)
    def _plus_one(self, n):
        return(function(self, n + 1))
    return(_plus_one)

class Spa(object):
    @plus_one
    def get_eggs(self, n = 2):
        return(n * 'eggs')

z       = Spa()
print(z.get_eggs(3))


class other2(object):
    def some_instancemethod(self, *args, **kwargs):
        print('self: %r' % self)
        print('args: %s' % pprint.pformat(args))
        print('kwargs: %s' % pprint.pformat(kwargs))

    @classmethod
    def some_classmethod(cls, *args, **kwargs):
        print('cls: %r' %cls)
        print('args: %s' % pprint.pformat(args))
        print('kwargs: %s' % pprint. pformat(kwargs))

    @staticmethod
    def some_staticmethod(*args, **kwargs):
        print('args: %s' % pprint.pformat(args))
        print('kwargs: %s' % pprint.pformat(kwargs))

o       = other2()

o.some_instancemethod(1, 2, a=3, b=4)
other2.some_instancemethod(1, 2, a=3, b=4)
o.some_classmethod(1, 2, a=3, b=4)
other2.some_classmethod()
other2.some_classmethod(1, 2, a=3, b=4)
o.some_staticmethod(1, 2, a=3, b=4)
other2.some_staticmethod(1, 2, a=3, b=4)

# decorators
class MoreSpam(object):
    def __init__(self, more=1):
        self.more   = more

    def __get__(self, instance, cls):
        return(self.more + instance.spam)

    def __set__(self, instance, value):
        instance.spam   = value - self.more

class Spamish(object):
    more_spam   = MoreSpam(5)

    def __init__(self, spam):
        self.spam   = spam

spamm   = Spamish(1)

print(spamm.spam)
print(spamm.more_spam)
spamm.more_spam  = 10
print(spamm.spam)
