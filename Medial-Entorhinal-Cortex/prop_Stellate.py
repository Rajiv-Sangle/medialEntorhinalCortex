import tensorflow as tf

C_s = 1.0  # uF/cm2

gNa = 52
ENa = 55

gK = 11
EK = -90

tau_ms = 0.15  # CONSTANT THROUGHOUT
gNaP = 0.5

gh = 1.5
Eh = -20

gL = 0.5
EL = -65

# Initial state of the Stellate Cell

V0 = -61.2  # mV

m0 = 0.0224224
h0 = 0.954963

n0 = 0.103519  # Check the 0

ms0 = 0.0678057

mhs0 = 0.118111
mhf0 = 0.0779264


def propNa(V):
    alpha_m = -0.1 * (V + 23) / (tf.exp(-0.1 * (V + 23)) - 1)

    beta_m = 4 * tf.exp(-(V + 48) / 18)

    alpha_h = 0.07 * tf.exp(-(V + 37) / 20)  # Note the minus sign

    beta_h = 1 / (tf.exp(-0.1 * (V + 7)) + 1)

    t_m = 1.0 / ((alpha_m + beta_m))  # * phi)
    t_h = 1.0 / ((alpha_h + beta_h))  # * phi)

    m_0 = alpha_m / (alpha_m + beta_m)
    h_0 = alpha_h / (alpha_h + beta_h)

    return m_0, t_m, h_0, t_h


def propK(V):
    alpha_n = -0.01 * (V + 27) / (tf.exp(-0.1 * (V + 27)) - 1)

    beta_n = 0.125 * tf.exp(-(V + 37) / 80)

    t_n = 1.0 / ((alpha_n + beta_n))  # * phi)
    n_0 = alpha_n / (alpha_n + beta_n)

    return n_0, t_n


def propNaP(V):
    ms_00 = 1 / (1 + tf.exp(-(V + 38) / 6.5))

    return ms_00, [tau_ms]*V.shape[0]


def propH(V):
    mhs_00 = 1 / (1 + (tf.exp((V + 2.83) / 15.9))) ** 58

    tau_mhs = 1 + 5.6 / (tf.exp((V - 1.7) / 14) + tf.exp(-(V + 260) / 43))

    mhf_00 = 1 / (1 + tf.exp((V + 79.2) / 9.78))

    tau_mhf = 1 + 0.51 / (tf.exp((V - 1.7) / 10) + tf.exp(-(V + 340) / 52))

    return mhs_00, tau_mhs, mhf_00, tau_mhf

def INaP(V, ms):
    return gNaP * ms * (V - ENa)


def Ih(V, mhs, mhf):
    return gh * (0.65 * mhf + 0.35 * mhs) * (V - Eh)

