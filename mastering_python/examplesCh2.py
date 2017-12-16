import logging
import warnings
#example import below
import djando.contrib.auth as auth_models
warnings.warn('Something deprecated', DeprecationWarning)

# not:
filter_modulo = lambda i, m: [i[j] for i in range(len(i))
        if i[j] % m]

# do
def filter_modulo(items, modulo):
    for item in items:
        if item % modulo:
            yield item

def fib(n):
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

'''
# catching all errors:
try: 
    value = int(user_input)
except Exception as e:
    logging.warn('Uncaught exception %r', e)
'''

'''
# instead of:
import io
if isinstance(f, io.IOBase):
    print("heeeeyyyyy")

simply do following
if hasattr(fh, 'read')
'''

# when comparing booleans or None, use "is"

'''
like:

for i, item in enumerate(my_list):
    do_something(i, item)

or

[do_something(i, item) for i, item in enumerate(my_list)]
'''
