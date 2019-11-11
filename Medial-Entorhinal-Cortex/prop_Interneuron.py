import tensorflow as tf

C_i = 1  # uF

gNa = 35
ENa = 55

gK = 9
EK = -90

gL = 0.1
EL = -65

# Initial conditions of Interneuron

V0 = -61.2
m0 = 0.0224224
h0 = 0.283859
n0 = 0.764751

def propNa(V):
    alpha_m = 0.1 * (V + 35) / (1 - tf.exp(-(V + 35) / 10))
    beta_m = 4 * tf.exp(-(V + 60) / 18)

    alpha_h = 0.07 * tf.exp(-(V + 58) / 20)
    beta_h = 1 / (tf.exp(-0.1 * (V + 28)) + 1)

    m_0 = alpha_m / (alpha_m + beta_m)
    t_m = 1 / (alpha_m + beta_m)

    h_0 = alpha_h / (alpha_h + beta_h)
    t_h = 1 / (alpha_h + beta_h)

    return m_0, t_m, h_0, t_h


def propK(V):
    alpha_n = 0.01 * (V + 34) / (1 - tf.exp(-0.1 * (V + 34)))
    beta_n = 0.125 * tf.exp(-(V + 44) / 80)

    n_0 = alpha_n / (alpha_n + beta_n)
    t_n = 1 / (alpha_n + beta_n)

    return n_0, t_n
