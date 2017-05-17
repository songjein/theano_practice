from theano import function, shared
import theano.tensor as T

state = shared(0)
inc = T.iscalar('inc')

accumulator = function([inc], state, updates=[(state, state+inc)])
decrementor = function([inc], state, updates=[(state, state-inc)])

# shared variables
# whose value may be shared between multiple functions
# its value is shared between many functions
# its value can be accessed and modified by the .get_value() and .set_value


# function's update param
# updates must be supplied with a list of pairs of the form (shared variable, new expression)


print (state.get_value())

accumulator(1)

print (state.get_value())

accumulator(300)

print (state.get_value())

state.set_value(-1)

print (state.get_value())

print "----------------------------------------"


# if you express some formula using a shared variable, but you do not want to use its value
# you can use the 'givens' param of function
# which replaces a particular node in a graph for the purpose of one particular function

fn_of_state = state * 2 + inc
# The type of foo must match the shared variable we are replacing
# with the ''givens''
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])

print state.get_value()

# we are using 3 for the state, not state.value
print skip_shared(1, 3)

# old state still there, but we didn't use it
print state.get_value()


# given is a mechanism that allows you to replace any part of your formula with a different expression 
