import theano
import theano.tensor as T

state = theano.shared(0)

inc = T.iscalar('inc')

accumulator = theano.function([inc], state, updates=[(state, state+inc)])

accumulator(10)

print state.get_value()

# Theano functions can be copied
# copy() method of function object
# The optimized graph of the original function is copied, so compilation only needs to be performed once
new_state = theano.shared(0)
new_accumulator = accumulator.copy(swap={state:new_state})

new_accumulator(100)
print state.get_value()
print new_state.get_value()


# create a copy with updates removed is possible
null_accumulator = accumulator.copy(delete_updates=True)

null_accumulator(100000)
print state.get_value()
