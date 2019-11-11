import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import transient_pulse as tp

def tf_check_type(t, y0):  # Ensure Input is Correct
    if not (y0.dtype.is_floating and t.dtype.is_floating):
        raise TypeError('Error in Datatype')


class _Tf_Integrator():

    def integrate(self, func, y0, t):
        time_delta_grid = t[1:] - t[:-1]

        def scan_func(y, t_dt):
            t, dt = t_dt
            dy = self._step_func(func, t, dt, y)  # Make code more modular.
            return y + dy

        y = tf.scan(scan_func, (t[:-1], time_delta_grid), y0)
        return tf.concat([[y0], y], axis=0)

    def _step_func(self, func, t, dt, y):
        k1 = func(y, t) #, t_con)
        half_step = t + dt / 2
        dt_cast = tf.cast(dt, y.dtype)  # Failsafe

        k2 = func(y + dt_cast * k1 / 2, half_step) #, t_con)
        k3 = func(y + dt_cast * k2 / 2, half_step) #, t_con)
        k4 = func(y + dt_cast * k3, t + dt) #, t_con)

        tp.idx = tp.idx + 1

        return tf.add_n([k1, 2 * k2, 2 * k3, k4]) * (dt_cast / 6)


def odeint(func, y0, t):

    t = tf.convert_to_tensor(t, preferred_dtype=tf.float64, name='t')
    y0 = tf.convert_to_tensor(y0, name='y0')
    tf_check_type(y0, t)
    return _Tf_Integrator().integrate(func, y0, t)
