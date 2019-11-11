def derivativeGate(x, x_00, tau_x):
    return -(x - x_00) / tau_x  # all gating variables follow first order dynamics

