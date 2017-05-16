import numpy
import theano.tensor as T
from theano import function, pp

# Adding Two Scalars
# Adding Two Scalars
# Adding Two Scalars

# define two symbols(variables objects)
# T.dscalar is the type we assign to '0-dim arrays(scalar) of doubles(d)"
# These are the instances of TensorVariables
x = T.dscalar('x')
y = T.dscalar('y')
print type(x)
print x.type
print T.dscalar
print x.type is T.dscalar

# By calling T.dscalar with a string argument, you create a Variable representing a floating-point scalar quantity
# with the given name


# z is another Variable, which represents the addition of x and y.
z = x + y
print pp(z) # pretty print


# function taking x and y as inputs and giving z as ouputs
# the first argument to function is a 'list of Variables' that will be provided as input to the function
# the second is a single Variable or a list of Variables 
f = function([x, y], z)

# f is compiled into C code!!
# the output of the function f is a numpy.ndarray with zero dimensions

print f(2,3)
print numpy.allclose(f(16.3, 12.1), 28.4)


# Adding Two Matrices
# Adding Two Matrices
# Adding Two Matrices

# dmatrix is the Type for matrices of doubles
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)

print f([[1,2], [3,4]], [[10,20], [30,40]])

# Variable is a NumPy array. We can use NumPy arrays directly as inputs
import numpy
print f(numpy.array([[1,2], [3,4]]), numpy.array([[10,20], [30,40]]))

