import theano
a = theano.tensor.vector()
out = a + a ** 10
f = theano.function([a], out)
print(f([0,1,2]))

a = theano.tensor.vector()
out = a[0] ** 2 + a[1] ** 2 + 2*a[0]*a[1]
f = theano.function([a], out)
print(f([2,4]))

a = theano.tensor.vector()  # declare variable
b = theano.tensor.vector()  # declare variable
out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
f = theano.function([a, b], out)   # compile function
print(f([1, 2], [4, 5]))  # prints [ 25.  49.]

