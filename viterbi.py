"""
implementation of the Viterbi algorithm
"""

import numpy as np
import operator


# based on:
# https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
def viterbi(start_prob, transition_prob, emission_prob, observations):
    """Return the best path, given an HMM model and a sequence of observations"""
    # A - initialise stuff
    n_samples = len(observations)
    n_states = transition_prob.shape[0]  # number of states
    c = np.zeros(n_samples)  # scale factors (necessary to prevent underflow)
    viterbi = np.zeros((n_states, n_samples))  # initialise viterbi table
    psi = np.zeros((n_states, n_samples))  # initialise the best path table
    best_path = np.zeros(n_samples).astype(int)  # this will be your output

    # B- appoint initial values for viterbi and best path (bp) tables - Eq (32a-32b)
    viterbi[:, 0] = start_prob.T * emission_prob[:, observations[0]].reshape(n_states)
    c[0] = 1.0 / np.sum(viterbi[:, 0])
    viterbi[:, 0] = c[0] * viterbi[:, 0]  # apply the scaling factor
    psi[0] = 0

    # C- Do the iterations for viterbi and psi for time>0 until T
    for t in range(1, n_samples):  # loop through time
        for s in range(0, n_states):  # loop through the states @(t-1)
            trans_p = viterbi[:, t - 1] * transition_prob[:, s]
            psi[s, t], viterbi[s, t] = max(enumerate(trans_p), key=operator.itemgetter(1))
            viterbi[s, t] = viterbi[s, t] * emission_prob[s, observations[t]]

        c[t] = 1.0 / np.sum(viterbi[:, t])  # scaling factor
        viterbi[:, t] = c[t] * viterbi[:, t]

    # D - Back-tracking
    best_path[n_samples - 1] = viterbi[:, n_samples - 1].argmax()  # last state
    for t in range(n_samples - 1, 0, -1):  # states of (last-1)th to 0th time step
        best_path[t - 1] = psi[best_path[t], t]

    return best_path
