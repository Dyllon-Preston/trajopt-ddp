from quadrotor import Quadrotor
from planning import *
import sympy as sp

mass = 1
Ixx = 0.005
Iyy = 0.005
Izz = 0.006
kt = 0.017
l = 0.2
g = 9.81
dt = 0.01

qr = Quadrotor(mass = mass,
               Ixx = Ixx,
               Iyy = Iyy,
               Izz = Izz,
               kt = kt,
               l = l,
               g = g,
               dt = dt)

print(qr)
breakpoint()

xg = np.zeros((12,1))
xg[:3] = np.array([[1],[1],[1]]) # FIX ME

Q = 1e-16*np.eye(12)
R = 1e-17*np.eye(4)

Qtf = 1e-14*np.eye(12)

l = lambda x, u: 0.5*((x - xg).T*Q*(x-xg) + u.T*R*u)
lf = lambda x, u: 0.5*(x - xg).T*Qtf*(x-xg)

ddp = DDP(
    dynamics = qr,
    l_func = l,
    lf_func = lf,
)

ddp.build_symbolic_gradients()

x0 = np.zeros((12,1))
x0[:3] = np.array([[-5],[-5],[-5]])

N = 100

ddp.run(x0, N)

breakpoint()