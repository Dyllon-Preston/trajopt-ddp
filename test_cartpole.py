from dynamics import *
from planning import *
import sympy as sp


mp = 0.1
mc = 1.0
l = 1.0
dt = 0.05

cart = CartPole(mp = mp,
               mc = mc,
               l = l,
               dt = dt)
print(cart)


cart.F(np.array([ 0.0000000e+00,  0.0000000e+00,  1.2246468e-16, -1.0000000e+00,        0.0000000e+00]), np.array([0.1]))



N = 250

x0 = np.array([0, 0, np.sin(np.pi), np.cos(np.pi), 0])
xg = np.array([0, 0, np.sin(0), np.cos(0), 0])

u0 = np.array([0])

Q = 10*np.eye(5)
Q[1,1] = Q[4,4] = 0
R = 0.01*np.eye(1)

Qtf = 1*np.eye(5)

l = lambda x, u, xg: (x - xg).T@Q@(x-xg) + u.T@R@u
lf = lambda x, u, xg: (x - xg).T@Qtf@(x-xg)

ddp = DDP(
    dynamics = cart,
    l_func = l,
    lf_func = lf,
)
ddp.build_symbolic_gradients()
x, u = ddp.run(x0, u0, xg, N, max_iter=300)

# cart.plot_states(x, u, dt)
cart.animate(x, u, dt)
