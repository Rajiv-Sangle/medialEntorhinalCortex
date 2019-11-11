# About: This code simulates an MEC motif

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Tf_Integrator

import prop_Interneuron as HI

import prop_Stellate as HS
from prop_Stellate import propH, propNaP

from prop_Common import propNa, propK

import func_math as fm


import transient_pulse as tp


# 1) Defining Synaptic Connectivity Parameters

n_n = 4  # num of neurons: 2 stellate + 2 interneurons
n_i = 2
n_s = 2

# Excitatory synapses

ext_mat = np.zeros((n_n, n_n))  # Excitatory Synapse Connectivity Matrix
ext_mat[0, 3] = 1
ext_mat[1, 2] = 1

## Parameters for Excitatory Synapse

n_ext = int(np.sum(ext_mat))  # num of excitatory synapses
alpha_ext = [100.0] * n_ext
beta_ext = [0.33] * n_ext

g_ext_si = [0.03] * n_n  # max conductance _si
E_ext = [0.0] * n_n  # Reversal Potential ... in mV

# Inhibitory synapses

inb_mat_ii = np.zeros((n_n, n_n))  # Mutual Inhibitory Synapse Connectivity Matrix
inb_mat_ii[0, 1] = 1
inb_mat_ii[1, 0] = 1

inb_mat_is = np.zeros((n_n, n_n))  # Interneuron-Stellate Synapse Connectivity Matrix
inb_mat_is[2, 0] = 1
inb_mat_is[3, 1] = 1

## Parameters for Inhibitory Synapse
n_inb_ii = int(np.sum(inb_mat_ii))  # num of mutual inhibitory synapses
n_inb_is = int(np.sum(inb_mat_is))  # num of interneuron-stellate synapses

alpha_inb = [3.33] * n_inb_ii  # or n_inb_is = 2
beta_inb = [0.11] * n_inb_ii

g_inb_ii = [0.6] * n_n  # max conductance _ii
g_inb_is = [1.0] * n_n  # max conductance _is

E_inb = [-80] * n_n  # Reversal Potential ... in mV

# 2) Defining Firing Thresholds
# F_t = [0.0] * n_n

'''
# 3) Defining Input Current as function of Time
def I_inj_t(t):
    return tf.constant(current_input.T, dtype=tf.float64)[tf.to_int32(t / epsilon)]
    # Turn indices to integer and extract from matrix
'''


# 4) Defining the currents

## Excitatory Synaptic Current

def I_ext(o, V):
    o_ = tf.Variable([0.0] * n_n ** 2, dtype=tf.float64)
    # print("HI\n")
    # print(o_.shape)
    # print(o.shape)
    ind = tf.boolean_mask(tf.range(n_n ** 2), ext_mat.reshape(-1) == 1)
    # print(ind.shape)
    o_ = tf.scatter_update(o_, ind, o)  # 1*(n_n*n_n) vector
    o_ = tf.transpose(tf.reshape(o_, (n_n, n_n)))  # n_n * n_ vector

    # ov_post = tf.Variable([0.0] * n_n ** 2, dtype=tf.float64)  # defined to capture the post-synaptic-voltage dependence
    # ov_post = tf.transpose(tf.transpose(o_) * (V - E_ext))  # defined to capture the post-synaptic-voltage dependence

    return tf.reduce_sum(tf.transpose((o_ * (V - E_ext)) * g_ext_si), 1)
    # return tf.reduce_sum(tf.transpose(ov_post * g_ext_si), 1)


## Mutual Inhibitory Synaptic Current

def I_inb_ii(o, V):
    o_ = tf.Variable([0.0] * n_n ** 2, dtype=tf.float64)
    ind = tf.boolean_mask(tf.range(n_n ** 2), inb_mat_ii.reshape(-1) == 1)
    o_ = tf.scatter_update(o_, ind, o)
    o_ = tf.transpose(tf.reshape(o_, (n_n, n_n)))

    ov_post = tf.transpose(tf.transpose(o_) * (V - E_inb))  # defined to capture the post-synaptic-voltage dependence
    return tf.reduce_sum(tf.transpose((o_ * (V - E_inb)) * g_inb_ii), 1)
    # return tf.reduce_sum(tf.transpose(ov_post * g_inb_ii), 1)


## Interneuron-Stellate Inhibitory Synaptic Current

def I_inb_is(o, V):
    o_ = tf.Variable([0.0] * n_n ** 2, dtype=tf.float64)
    ind = tf.boolean_mask(tf.range(n_n ** 2), inb_mat_is.reshape(-1) == 1)
    o_ = tf.scatter_update(o_, ind, o)
    o_ = tf.transpose(tf.reshape(o_, (n_n, n_n)))

    ov_post = tf.transpose(tf.transpose(o_) * (V - E_inb))  # defined to capture the post-synaptic-voltage dependence

    return tf.reduce_sum(tf.transpose((o_ * (V - E_inb)) * g_inb_is), 1)
    # return tf.reduce_sum(tf.transpose(ov_post * g_inb_is), 1)


## Other Currents

def INa(V, m, h):
    return gNa * m ** 3 * h * (V - ENa)


def IK(V, n):
    return gK * n ** 4 * (V - EK)


def IL(V):
    return gL * (V - EL)


'''
# 5) Defining the gating variables

def propNa(V):
    m_0 = tf.Variable([0.0] * n_n, dtype=tf.float64)
    t_m = tf.Variable([0.0] * n_n, dtype=tf.float64)

    h_0 = tf.Variable([0.0] * n_n, dtype=tf.float64)
    t_h = tf.Variable([0.0] * n_n, dtype=tf.float64)

    m_0[:2], t_m[:2], h_0[:2], t_h[:2] = HI.propNa(V[:2])

    m_0[2:], t_m[2:], h_0[2:], t_h[2:] = HS.propNa(V[2:])

    return m_0, t_m, h_0, t_h


def propK(V):
    n_0 = tf.Variable([0.0] * n_n, dtype=tf.float64)
    t_n = tf.Variable([0.0] * n_n, dtype=tf.float64)

    n_0[:2], t_n[:2] = HI.propK(V[:2])

    n_0[2:], t_n[2:] = HS.propK(V[2:])

    return n_0, t_n
'''


# 6) Derivative of the State Vector

def dXdt(X, t):

    V = X[: 1 * n_n]
    # print("Hiii there\n")
    # print(V.shape)
    m = X[1 * n_n: 2 * n_n]
    h = X[2 * n_n: 3 * n_n]

    n = X[3 * n_n: 4 * n_n]

    ms = X[4 * n_n: 5 * n_n]

    mhs = X[5 * n_n: 6 * n_n]
    mhf = X[6 * n_n: 7 * n_n]

    o_ext = X[7 * n_n: 7 * n_n + n_ext]

    o_inb_ii = X[7 * n_n + n_ext: 7 * n_n + n_ext + n_inb_ii]

    #o_inb_is = X[7 * n_n + n_ext + n_inb_ii: 7 * n_n + n_ext + n_inb_ii + n_inb_is]
    o_inb_is = X[7 * n_n + n_ext + n_inb_ii:]

    # print(o_inb_ii.shape)
    #I_pulse = X[7 * n_n + n_ext + n_inb_ii + n_inb_is:]

    # fire_t = X[-n_n:]  # Last n_n values are the last fire times as updated by the modified integrator

    # dVdt = tf.Variable([0.0] * n_n, dtype=tf.float64)

    I_syn = I_ext(o_ext, V) + I_inb_ii(o_inb_ii, V) + I_inb_is(o_inb_is, V)
    #I_syn = I_inb_ii(o_inb_ii, V) + I_inb_is(o_inb_is, V)
    # I_syn = 0
    '''
    dVdt[:2] = (I_inj_t(t) - INa(V, m, h) - IK(V, n) - IL(V) - I_syn) / C_m[:n_i]
    dVdt[2:] = (I_inj_t(t) - INa(V, m, h) - IK(V, n) - IL(V) - I_syn - HS.INaP(V, ms) - HS.Ih(V, mhs, mhf)) / C_m[-n_s:]
    
    dVdt[:2] = (current_input[:n_i] - INa(V, m, h) - IK(V, n) - IL(V) - I_syn) / C_m[:n_i]
    dVdt[2:] = (current_input[-n_s:] - INa(V, m, h) - IK(V, n) - IL(V) - I_syn - HS.INaP(V, ms) - HS.Ih(V, mhs, mhf)) \
               / C_m[-n_s:]
    '''

    I_theta = A * tf.sin(2 * 3.14 * omega * t + phi) * (V - V_th)
    #I_theta = 0

    I_pulse = tf.Variable([0.0]*n_n, dtype=tf.float64)
    '''
    with tf.Session() as sess1:
        tf.global_variables_initializer().run()
        t_ = sess1.run(t)
    '''
    #t_ = tf.constant(t_lim)
    #I_pulse = tf.cond(200 < t_lim <= 250, lambda: tf.constant([0, 1.2, 0, 0], dtype=tf.float64), lambda: tf.constant([0, 0, 0, 0], dtype=tf.float64))
    #I_pulse = tf.cond(200 < t <= 250, lambda: [0, 1.2, 0, 0], lambda: [0, 0, 0, 0])
    '''
    if n_n>200:
        I_pulse = tf.constant([0, 1.2, 0, 0], dtype=tf.float64)
    else:
        I_pulse = tf.constant([0, 0, 0, 0], dtype=tf.float64)
    '''

    #if 200 < tp.t_[tp.idx] <=250 :
    #    I_pulse = tf.constant([0.0, 1.4, 0.0, 0.0], dtype=tf.float64)

    dVdt = (current_input + I_dc + I_pulse - IL(V) - IK(V, n) - INa(V, m, h) - I_syn - HS.INaP(V, ms) - HS.Ih(V, mhs, mhf) - I_theta) / C_m

    m_0, t_m, h_0, t_h = propNa(V)
    n_0, t_n = propK(V)

    ms_0, t_ms = propNaP(V)
    mhs_0, t_mhs, mhf_0, t_mhf = propH(V)

    dmdt = fm.derivativeGate(m, m_0, t_m)
    dhdt = fm.derivativeGate(h, h_0, t_h)

    dndt = fm.derivativeGate(n, n_0, t_n)

    '''
    dmsdt = tf.Variable([0.0] * n_n, dtype=tf.float64)
    dmsdt[2:] = fm.derivativeGate(ms, ms_0, t_ms)


    dmsdt = [0.0] * n_n
    dmsdt[2:] = fm.derivativeGate(ms, ms_0, t_ms)
    '''

    dmsdt = tf.concat([tf.constant([0.0] * 2, dtype=tf.float64), fm.derivativeGate(ms[2:], ms_0[2:], t_ms[2:])], 0)

    dmhsdt = tf.concat([tf.constant([0.0] * 2, dtype=tf.float64), fm.derivativeGate(mhs[2:], mhs_0[2:], t_mhs[2:])], 0)

    dmhfdt = tf.concat([tf.constant([0.0] * 2, dtype=tf.float64), fm.derivativeGate(mhf[2:], mhf_0[2:], t_mhf[2:])], 0)

    ## Updation for S_ext
    #V_ = tf.reduce_sum(tf.transpose(ext_mat) * V, 0)
    #V_ = tf.reduce_sum(ext_mat * V, 1)
    #V_ = tf.matmul(tf.constant(ext_mat, dtype=tf.float64), tf.reshape())
    #V_ = tf.matmul(ext_mat, V)
    #V_ = V
    V_ = V
    F_ext = (1 + tf.tanh(V_ / 4)) / 2  # 4x1
    F_ext = tf.multiply(tf.constant(ext_mat, dtype=tf.float64), F_ext)  # 4x4
    F_ext = tf.boolean_mask(tf.reshape(F_ext, (-1,)), ext_mat.reshape(-1) == 1)

    do_extdt = alpha_ext * (1 - o_ext) * F_ext - beta_ext * o_ext

    ## Updation for S_ii
    V_ = tf.reduce_sum(tf.transpose(inb_mat_ii) * V, 0)
    #V_ = tf.reduce_sum(inb_mat_ii * V, 1)
    V_ = V
    F_ext = (1 + tf.tanh(V_ / 4)) / 2  # 4x1
    F_ext = tf.multiply(tf.constant(inb_mat_ii, dtype=tf.float64), F_ext)
    F_ext = tf.boolean_mask(tf.reshape(F_ext, (-1,)), inb_mat_ii.reshape(-1) == 1)

    do_inb_iidt = alpha_inb * (1 - o_inb_ii) * F_ext - beta_inb * o_inb_ii

    ## Updation for S_is
    V_ = tf.reduce_sum(tf.transpose(inb_mat_is) * V, 0)
    #V_ = tf.reduce_sum(inb_mat_is * V, 1)
    V_ = V
    F_ext = (1 + tf.tanh(V_ / 4)) / 2  # 4x1
    F_ext = tf.multiply(tf.constant(inb_mat_is, dtype=tf.float64), F_ext)
    F_ext = tf.boolean_mask(tf.reshape(F_ext, (-1,)), inb_mat_is.reshape(-1) == 1)

    do_inb_isdt = alpha_inb * (1 - o_inb_is) * F_ext - beta_inb * o_inb_is

    '''
    do_extdt = [0.0] * n_n
    do_inb_iidt = [0.0] * n_n
    do_inb_isdt = [0.0] * n_n
    ## Updation for fire times ##

    # dfdt = tf.zeros(tf.shape(fire_t), dtype=fire_t.dtype)  # zero change in fire_t
    
    # current = tf.concat([dVdt, dmdt, dhdt, dndt, dmsdt, dmhsdt, dmhfdt, do_extdt, do_inb_iidt, do_inb_isdt, dfdt], 0)
    current = tf.concat([dVdt, dmdt, dhdt, dndt, dmsdt, dmhsdt, dmhfdt, do_extdt, do_inb_iidt, do_inb_isdt], 0)
    '''

    current = tf.concat([dVdt, dmdt, dhdt, dndt, dmsdt, dmhsdt, dmhfdt, do_extdt, do_inb_iidt, do_inb_isdt], 0)
    return current


# 7) Initial Parameters

C_m = 1.0  # tf.Variable([1.0] * n_n, dtype=tf.float64)

gNa = [0.0] * n_n
gNa[:2] = [HI.gNa] * 2
gNa[2:] = [HS.gNa] * 2

ENa = [0.0] * n_n
ENa[:2] = [HI.ENa] * 2
ENa[2:] = [HS.ENa] * 2

gK = [0.0] * n_n
gK[:2] = [HI.gK] * 2
gK[2:] = [HS.gK] * 2

EK = [0.0] * n_n
EK[:2] = [HI.EK] * 2
EK[2:] = [HS.EK] * 2

gL = [0.0] * n_n
gL[:2] = [HI.gL] * 2
gL[2:] = [HS.gL] * 2

EL = [0.0] * n_n
EL[:2] = [HI.EL] * 2
EL[2:] = [HS.EL] * 2

I_dc = [1.0, 1.0, 0, 0]

A = [0.0] * n_n
A[:2] = [0.04] * n_i

omega = 8  # Hz

phi = [0]*n_n
#phi[1] = 20

V_th = [-80] * n_n

P_min = -0.05  # uA/cm2
P_max = 1.00

T = 30  # second
t_delta = 1/omega

tau_rise = 2.0  # ms
tau_fall = 2.0  # ms



# 8) Initial State Vector
V0 = [-61.2] * n_n

m0 = [0.0] * n_n
m0[:2] = [HI.m0] * n_i
m0[2:] = [HS.m0] * n_s

h0 = [0.0] * n_n
h0[:2] = [HI.h0] * n_i
h0[2:] = [HS.h0] * n_s

n0 = [0.0] * n_n
n0[:2] = [HI.n0] * n_i
n0[2:] = [HS.n0] * n_s

ms0 = [0.0] * n_n
ms0[2:] = [HS.ms0] * n_s

mhs0 = [0.0] * n_n
mhs0[2:] = [HS.mhs0] * n_s

mhf0 = [0.0] * n_n
mhf0[2:] = [HS.mhf0] * n_s

o_ext0 = [0.0] * n_ext
o_inb_ii = [0.0] * n_inb_ii
o_inb_is = [0.0] * n_inb_is

# y0 = tf.constant([V0 + m0 + h0 + n0 + ms0 + mhs0 + mhf0 + o_ext0 + o_inb_ii + o_inb_is], dtype=tf.float64)  # Note!!!
y0 = tf.constant(V0 + m0 + h0 + n0 + ms0 + mhs0 + mhf0 + o_ext0 + o_inb_ii + o_inb_is, dtype=tf.float64)

#epsilon = 0.01
#t = np.arange(0, 500, epsilon)
t = tp.t_

t_start = np.arange(0, 200, T)
t_end = t_start + t_delta


current_input = [0.0] * n_n
current_input[:n_i] = [0.2] * n_i
current_input[n_i:] = [-2.71] * n_s

current_input = tf.constant(current_input, dtype=tf.float64)

state = Tf_Integrator.odeint(dXdt, y0, t)  # , n_n, F_t)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    state = sess.run(state)
    #wave = 0.04*tf.sin(2*3.14*omega*t)


plt.figure(figsize=(12, 3))

plt.plot(t, state[:, 0], label="$X_1 Interneuron $")
plt.plot(t, state[:, 2], label="$X_2 Stellate $")
#plt.plot(t, state[:, 1], label="$X_3 Stellate $")
#plt.plot(t, state[:, 3], label="$X_4 Stellate $")
#plt.plot(t, wave[:, 0], label="$X_4 Stellate $")
plt.title("Medial Enthorinal Cortex Motif")  # $X_1$ --> $X_2$ --◇ $X_3$")
plt.ylim([-90, 60])
plt.xlabel("Time (in ms)")
plt.ylabel("Voltage (in mV)")
plt.legend()

plt.tight_layout()
plt.show()
'''

#plt.figure(figsize=(12, 17))
for i in range(0,4,2):
    plt.subplot(10, 1, i + 1)
    plt.plot(t, state[:, i])
    plt.title("MEC motif= {:0.1f}".format(i+1))
    plt.ylim([-90, 60])
    plt.xlabel("Time (in ms)")
    plt.ylabel("Voltage (in mV)")
'''

#plt.tight_layout()
plt.show()

#'''