import tensorflow as tf

import prop_Interneuron as pI
import prop_Stellate as pS


def propNa(V):
    m0_1, t_m1, h0_1, t_h1 = pI.propNa(V[:2])
    m0_2, t_m2, h0_2, t_h2 = pS.propNa(V[2:])

    return tf.concat([m0_1, m0_2], 0), tf.concat([t_m1, t_m2], 0), tf.concat([h0_1, h0_2], 0), tf.concat([t_h1, t_h2], 0)


def propK(V):
    n0_1, t_n1 = pI.propK(V[:2])
    n0_2, t_n2 = pS.propK(V[2:])

    return tf.concat([n0_1, n0_2], 0), tf.concat([t_n1, t_n2], 0)
