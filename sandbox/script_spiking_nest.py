'''
A simple nest script.
Nest is a mess to install but at least it has a lot of examples to find documentation.
This code is working.

TODO:
- Understand the relation ship between the parameters of the tf model (W and b) and this nest model.
- Make a function that takes as input the n_input, ... and tau_m and returns spike_x,spike_y and spike_z

'''


import nest
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
b0 = 1

W_hid = rd.randn(n_in,n_hid) / np.sqrt(n_in) * sigma
W_out = rd.randn(n_hid,n_out) / np.sqrt(n_hid) * sigma

# Time constants in ms
dt = 1.
tau_m = 30.
tau_syn = 1.
T = 1000.

V0 = -1.
W0 = 1 * np.abs(V0)

v_reset = v_rest = V0
v_threshold = V0 - V0 * b0

neuron_model = "iaf_neuron"
neuron_params = {"C_m": 1.,
                 "tau_m": tau_m,
                 "t_ref": 1.,
                 "E_L": 0.0,
                 "V_reset": v_reset,
                 "V_m": 0.0,
                 "V_th": v_threshold}

initial_values={'v': v_reset}
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})

# Load dataset
train_prop = .5
x_train,x_test,z_train,z_test = ff.load_small_digits(train_prop,n_out)

# Neuron populations
x0 = x_train[0,:]
p_in = nest.Create("spike_generator", n_in)
for k in range(n_in):
    if x0[k] > 1.: nest.SetStatus((p_in[k],),{'spike_times': [1.]})

p_hid = nest.Create(neuron_model, n_hid, params=neuron_params)
p_out = nest.Create(neuron_model, n_out, params=neuron_params)

spikedetector_in = nest.Create("spike_detector",params={"withgid": True, "withtime": True})
spikedetector_hid = nest.Create("spike_detector",params={"withgid": True, "withtime": True})
spikedetector_out = nest.Create("spike_detector",params={"withgid": True, "withtime": True})

# Making connections
def W_to_connector(W,w0=1,delay=1):
    n_i,n_j = W.shape
    l = []
    for i in range(n_i):
        for j in range(n_j):
            l.append((i,j,W[i,j] * w0,delay))
    con = sim.FromListConnector(l, column_names=["weight", "delay"])

    return con

conn_dict = {"rule": "all_to_all"}
syn_dict = {"delay": 1., "weight": W0}
nest.Connect(p_in, p_hid, conn_dict, syn_dict)
nest.Connect(p_hid, p_out, conn_dict, syn_dict)

# Record spikes
nest.Connect(p_in,spikedetector_in)
nest.Connect(p_hid,spikedetector_hid)
nest.Connect(p_out,spikedetector_out)

# Run the simulation
nest.Simulate(T)

# plot the spike trains
fig,ax_list = plt.subplots(3)

for k,spike_detector,ax in zip(range(3),[spikedetector_in,spikedetector_hid,spikedetector_out],ax_list):
    dSD = nest.GetStatus(spike_detector, keys='events')[0]
    evs = dSD["senders"]
    ts = dSD["times"]
    ax.scatter(ts,evs)
    ax.set_xlim([0,T])

plt.show()