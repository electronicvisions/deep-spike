'''
Simple script to run a basic pyNN network.
I actually gave up because pyNN is poorly implemented.
I could not understand how to generate a spike a some chosen time neither how to record the spikes.

'''


import pyNN.nest as sim
import numpy.random as rd
import numpy as np
import utils as ff
import matplotlib.pyplot as plt

# Parameters
n_in = 8*8
n_hid = 192
n_out = 10

# Neuron parameters as in ANN model
sigma = 1.
b0 = 3

W_hid = rd.randn(n_in,n_hid) / np.sqrt(n_in) * sigma
W_out = rd.randn(n_hid,n_out) / np.sqrt(n_hid) * sigma

# Time constants in ms
dt = 1.
tau_m = 30.
tau_syn = 1.
T = 1000.

V0 = -65.0
v_reset = v_rest = V0
v_threshold = V0 + np.abs(V0) * b0

neuron_model = sim.IF_curr_exp
cell_params = {
    'tau_refrac': 1.,
    'v_thresh': v_threshold,
    'tau_m': tau_m,
    'tau_syn_E': tau_syn,
    'v_rest': v_rest,
    'cm': 1.,
    'v_reset': v_reset,
    'tau_syn_I': tau_syn,
    'i_offset': 0.}

initial_values={'v': v_reset}

# Load dataset
train_prop = .5
x_train,x_test,z_train,z_test = ff.load_small_digits(train_prop,n_out)
bin_x = x_train[0,:] > .2
ll = np.zeros(n_in,dtype=object)
ll[bin_x] = [1.]

# Neuron populations
p_in = sim.Population(n_in, sim.SpikeSourcePoisson, cellparams={'rate': 100})
p_hid = sim.Population(n_hid, neuron_model, cellparams=cell_params,initial_values=initial_values)
p_out = sim.Population(n_out, neuron_model,cellparams=cell_params,initial_values=initial_values)

# Making connections
def W_to_connector(W,w0=1,delay=1):
    n_i,n_j = W.shape
    l = []
    for i in range(n_i):
        for j in range(n_j):
            l.append((i,j,W[i,j] * w0,delay))
    con = sim.FromListConnector(l, column_names=["weight", "delay"])

    return con

connections_hid = sim.Projection(p_in, p_hid, W_to_connector(W_hid))
connections_out = sim.Projection(p_hid, p_out, W_to_connector(W_out))

# Record spikes
p_in.record('spikes')
p_hid.record('spikes')
p_out.record('spikes')

t = sim.run(T)


# Get recorded data and plot
data_block = p_hid.get_data()
print(data_block)

fig,ax_list = plt.subplots(3)

for k,p,ax in zip(range(3),[p_in,p_hid,p_out],ax_list):
    ax.scatter(p.get_data()['spikes'])
    ax.set_xlim[0,T]


plt.show()