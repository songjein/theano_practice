from theano import In
from theano import function
import theano.tensor as T

x, y = T.dscalars('x', 'y')

z = x + y

f = function([x, In(y, value=1)], z)

# In class allows you to specify properties of your function's params with greater detail

print f(33)
print f(33,2)

x, y, w = T.dscalars('x', 'y', 'w')

z = (x + y) * w

# The symbolic variable objects(ex. discalar) have name attributes 
# and these are the names of the keyword params in the functions
# 
# We can override the symbolic variable's name attribute with a name to be used for this function
f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)

print f(33)
print f(33, 2)
print f(33, 0, 1)
print f(33, w_by_name=1)
print f(33, w_by_name=1, y=0)
