import theano
import theano.tensor as T
import numpy as np


# we will create a set of states, the agent get a reward for getting to the 5th one(4 in zero based array).
# the agent can go forward or backward by one state with wrapping(so if you go back from the 1st state you go to the
# end).
states = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
NUM_STATES = len(states)
NUM_ACTIONS = 2
FUTURE_REWARD_DISCOUNT = 0.5
LEARNING_RATE = 0.1


def hot_one_state(index):
    array = np.zeros(NUM_STATES)
    array[index] = 1.
    return array.reshape(array.shape[0], 1)  # Theano is sad if the shape looks like (10,) rather than (10,1)


state, targets = T.dmatrices('state', 'targets')
hidden_weights = theano.shared(value=np.zeros((NUM_ACTIONS, NUM_STATES)), name='hidden_weights')

output_fn = T.dot(hidden_weights, state)
output = theano.function([state], output_fn)

states_input = T.dmatrix('states_input')
loss_fn = T.mean((T.dot(hidden_weights, states_input) - targets) ** 2)

gradient = T.grad(cost=loss_fn, wrt=hidden_weights)

train_model = theano.function(
    inputs=[states_input, targets],
    outputs=loss_fn,
    updates=[[hidden_weights, hidden_weights - LEARNING_RATE * gradient]],
    allow_input_downcast=True
)

for i in range(50):
    state_batch = []
    rewards_batch = []

    # create a batch of states
    for state_index in range(NUM_STATES):
        state_batch.append(hot_one_state(state_index)[:,0])

        minus_action_index = (state_index - 1) % NUM_STATES
        plus_action_index = (state_index + 1) % NUM_STATES

        minus_action_state_reward = output(state=hot_one_state(minus_action_index))
        plus_action_state_reward = output(state=hot_one_state(plus_action_index))

        # these action rewards are the results of the Q function for this state and the actions minus or plus
        action_rewards = [states[minus_action_index] + FUTURE_REWARD_DISCOUNT * np.max(minus_action_state_reward),
                          states[plus_action_index] + FUTURE_REWARD_DISCOUNT * np.max(plus_action_state_reward)]
        rewards_batch.append(action_rewards)

    train_model(state_batch, np.array(rewards_batch).T)
    print([states[x] + np.max(output(hot_one_state(x))) for x in range(NUM_STATES)])