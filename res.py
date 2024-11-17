import numpy as np
def test(_Dummy_62, _Dummy_63, xg):
    breakpoint()
    [x, sin_theta, cos_theta, xdot, thetadot] = _Dummy_62
    [u] = _Dummy_63
    return np.array([[200.0*x - 200.0*xg[0, 0], 200.0*sin_theta - 200.0*xg[1, 0], 200.0*cos_theta - 200.0*xg[2, 0], 200.0*xdot - 200.0*xg[3, 0], 200.0*thetadot - 200.0*xg[4, 0]]])
