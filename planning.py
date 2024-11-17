import numpy as np
import sympy as sym
from scipy.integrate import solve_ivp

from res import test
import sys
import time

class Logger:
    HEADER = "\033[38;5;214m"  # Orange for the header
    BORDER = "\033[37m"        # Light gray for border and separator
    TITLE = "\033[95m"         # Pink for section titles
    VALUE = "\033[97m"         # White for values
    RESET = "\033[0m"          # Reset to default color

    @staticmethod
    def start_ddp_algorithm():
        print(
            f"{Logger.BORDER}╔═══════════════════════════════════════════════╗\n"
            f"║             {Logger.HEADER}Starting DDP Algorithm{Logger.BORDER}            ║\n"
            f"╚═══════════════════════════════════════════════╝{Logger.RESET}"
        )

    @staticmethod
    def log_iteration(i, max_iter, J, alpha):
        
        

        # Print the updated iteration status
        print(
            f"{Logger.BORDER}╔═══════════════════════════════════════════════╗\n"
            f"║ {Logger.TITLE}Itr:{Logger.RESET} {Logger.VALUE}{i:>4}/{max_iter:<4}{Logger.RESET}  "
            f"{Logger.TITLE}J:{Logger.RESET} {Logger.VALUE}{J:<12.6f}{Logger.RESET}  "
            f"{Logger.TITLE}\u03B1:{Logger.RESET} {Logger.VALUE}{alpha:.3E}{Logger.RESET} ║\n"
            f"╚═══════════════════════════════════════════════╝{Logger.RESET}"
        )
        # Move the cursor up by 3 lines to overwrite the previous iteration output
        sys.stdout.write("\033[A" * 3)

        sys.stdout.flush()

    @staticmethod
    def end_ddp_algorithm():
        # Move the cursor up by 3 lines to overwrite the previous iteration output
        sys.stdout.write("\033[B" * 3)
        print(
            f"{Logger.BORDER}╔═══════════════════════════════════════════════╗\n"
            f"║             {Logger.HEADER}DDP Algorithm Complete!{Logger.BORDER}           ║\n"
            f"╚═══════════════════════════════════════════════╝{Logger.RESET}"
        )





class DDP():
    def __init__(
            self,
            dynamics,
            l_func,
            lf_func,
            alphas = 1.1 ** (-np.arange(10) ** 2),
            hessians = False):
        self.dynamics = dynamics
        self.l = l_func
        self.lf = lf_func
        self.alphas = alphas
        self.hessians = hessians

        # self.x_symbolic = self.dynamics.get_symbolic_states()
        # self.u_symbolic = self.dynamics.get_symbolic_controls()
        # self.F_symbolic = self.dynamics.get_symbolic_discrete_dynamics()


    def build_symbolic_gradients(self):

        Nx = 5
        Nu = 1



        x = sym.Matrix(sym.symbols(f'x:{Nx}'))
        u = sym.Matrix(sym.symbols(f'u:{Nu}'))
        xg = sym.Matrix(sym.symbols(f'xg:{Nx}'))


        l = self.l(x, u, xg)

        lx = l.jacobian(x)
        lu = l.jacobian(u)
        lxx = lx.jacobian(x)
        luu = lu.jacobian(u)
        lux = lu.jacobian(x)


        lf = self.lf(x, u, xg)

        lfx = lf.jacobian(x)
        lfxx = lfx.jacobian(x)

        F = self.dynamics.F(x, u)

        Fx  = F.jacobian(x)
        Fu  = F.jacobian(u)
        Fxx = [Fx.row(i).jacobian(x).tolist() for i in range(x.shape[0])]
        Fuu = [Fu.row(i).jacobian(u).tolist() for i in range(x.shape[0])]
        Fux = [Fu.row(i).jacobian(x).tolist() for i in range(x.shape[0])]

        # self.Fx  = sym.lambdify(args = [x, u], expr = Fx, cse = True)
        # self.Fu  = sym.lambdify(args = [x, u], expr = Fu, cse = True)
        # self.Fxx = sym.lambdify(args = [x, u], expr = Fxx, cse = True)
        # self.Fuu = sym.lambdify(args = [x, u], expr = Fuu, cse = True)
        # self.Fux = sym.lambdify(args = [x, u], expr = Fux, cse = True)
        

        self.F = sym.lambdify(args = [x, u], expr = F, cse = True)

        self.lfx  = sym.lambdify(args = [x, u, xg], expr = lfx, cse = True)
        self.lfxx = sym.lambdify(args = [x, u, xg], expr = lfxx, cse = True)
        self.gradients = sym.lambdify(args = [x, u, xg], expr = [lx, lu, lxx, luu, lux, Fx, Fu, Fxx, Fuu, Fux], cse = True)


        # Vx =  sym.MatrixSymbol('Vx', x.shape[0], 1)
        # Vx_array = sym.Array([Vx[i] for i in range(Vx.shape[0])])
        # Vxx = sym.MatrixSymbol('Vxx', x.shape[0], x.shape[0])

  
        # Qx = sym.Matrix(lx + Fx.T@Vx)
        # Qu = sym.Matrix(lu + Fu.T@Vx)
        # Qxx = lxx + Fx.T@Vxx@Fx + sym.Matrix(sym.tensorcontraction(sym.tensorproduct(Vx_array, Fxx), (0,1)))
        # Quu = luu + Fu.T@Vxx@Fu + sym.Matrix(sym.tensorcontraction(sym.tensorproduct(Vx_array, Fuu), (0,1)))
        # Qux = lux + Fu.T@Vxx@Fx + sym.Matrix(sym.tensorcontraction(sym.tensorproduct(Vx_array, Fux), (0,1)))

        

        # jac = F.jacobian(x)
        # self.fxx = sym.lambdify(args = [x, u], expr = [jac.row(i).jacobian(x) for i in range(x.shape[0])])
        # jac = F.jacobian(u)
        # self.fuu = sym.lambdify(args = [x, u], expr = [jac.row(i).jacobian(u) for i in range(x.shape[0])])

    






        # Qxx = sym.Matrix([[sym.Matrix([lxx[i,j]]) +  Fx[:,i].T*Vxx*Fx[:,j] + Vx.T*sym.Derivative(sym.Derivative(F, xi), xj).doit() for j, xj in enumerate(x)] for i, xi in enumerate(x)])
        # Quu = sym.Matrix([[sym.Matrix([luu[i,j]]) +  Fu[:,i].T*Vxx*Fu[:,j] + Vx.T*sym.Derivative(sym.Derivative(F, ui), uj).doit() for j, uj in enumerate(u)] for i, ui in enumerate(u)])
        # Qux = sym.Matrix([[sym.Matrix([lux[i,j]]) +  Fu[:,i].T*Vxx*Fx[:,j] + Vx.T*sym.Derivative(sym.Derivative(F, ui), xj).doit() for j, xj in enumerate(x)] for i, ui in enumerate(u)]) # CHECK ME, EsymECIALLY Lux[j,i] INDEX

        # lfx = lf.jacobian(x)
        # lfxx = lfx.jacobian(x)

        # self.F = sym.lambdify(args = [x, u], expr = F, cse = True)

        # self.Qx = sym.lambdify(args = [x, u, Vx], expr = Qx, cse = True)
        # self.Qu = sym.lambdify(args = [x, u, Vx], expr = Qu, cse = True)
        # # self.Qxx = sym.lambdify(args = [x, u, Vx, Vxx], expr = Qxx, cse = True)
        # self.Quu = sym.lambdify(args = [x, u, Vx, Vxx], expr = Quu, cse = True)
        # self.Qux = sym.lambdify(args = [x, u, Vx, Vxx], expr = Qux, cse = True)

        # self.Qs = sym.lambdify(args = [x, u, Vx, Vxx], expr = [Qx, Qu, Qxx, Quu, Qux], cse = True)

        # self.l = sym.lambdify(args = [x, u, xg], expr = l, cse = True)
        # self.lf = sym.lambdify(args = [x, u, xg], expr = lf, cse = True)

        # self.lfx = sym.lambdify(args = [x, u, xg], expr = lfx, cse = True)
        # self.lfxx = sym.lambdify(args = [x, u, xg], expr = lfxx, cse = True)

    def cost(self, x, u, xg):
        J = 0
        for i in range(len(u)):
            J += self.l(x[i],u[i], xg)
        J += self.lf(x[-1],u[-1], xg)
        return J
        

        

    def run(self, x0, u0, xg, N, max_iter = 200):

        Logger.start_ddp_algorithm()


        x = np.zeros((N + 1, len(x0)))
        x[0] = x0
        # u = np.random.uniform(-1.0, 1.0, (N, u0.shape[0]))
        u = np.ones((N, u0.shape[0]))*0.1

        k = np.zeros((x.shape[0], u.shape[1]))
        K = np.zeros((x.shape[0], u.shape[1], x.shape[1]))

        x, _ = self.forward_pass(x, u, k, K)
        for i in range(max_iter):
            
            k, K = self.backward_pass(x, u, xg)

            J = self.cost(x, u, xg)
            

            J = self.cost(x, u, xg)
            for j, alpha in enumerate(self.alphas):
                xi, ui = self.forward_pass(x, u, alpha*k, K)
                
                Ji = self.cost(xi, ui, xg)
                if Ji < J:
                    
                    J = Ji
                    x, u = xi, ui
                    
                
            # print(F'Iteration: {i}/{max_iter}.   Cost: {J}.   Line Search Iterations: {j}')  

            Logger.log_iteration(i+1, max_iter, J, self.alphas[j])

        Logger.end_ddp_algorithm()
        
        return x, u


    def forward_pass(self, x0, u0, k, K):
        x = np.zeros((x0.shape[0], x0.shape[1]))
        x[0] = x0[0]
        u = np.zeros((u0.shape[0], u0.shape[1]))
        for i in range(x0.shape[0] - 1):
            u[i] = u0[i] + k[i] + K[i]@(x[i] - x0[i])
            x[i+1] = self.F(x[i], u[i]).flatten()
        return x, u


    def backward_pass(self, x, u, xg):

        k = np.zeros((x.shape[0], u.shape[1]))
        K = np.zeros((x.shape[0], u.shape[1], x.shape[1]))

        

        Vx = np.zeros((x.shape[0], x.shape[1]))
        Vxx = np.zeros((x.shape[0], x.shape[1], x.shape[1]))

        Vx = self.lfx(x[-1],u[-1], xg).flatten()
        Vxx = self.lfxx(x[-1],u[-1], xg)
    
        for i in range(x.shape[0] - 2, -1, -1):
            lx, lu, lxx, luu, lux, Fx, Fu, Fxx, Fuu, Fux = self.gradients(x[i], u[i], xg)
            lx = lx.flatten()
            lu = lu.flatten()


            Qx = lx + Fx.T@Vx
            Qu = lu + Fu.T@Vx
            Qxx = lxx + Fx.T@Vxx@Fx
            Quu = luu + Fu.T@Vxx@Fu
            Qux = lux + Fu.T@Vxx@Fx

            if self.hessians:
                Qxx += np.tensordot(Vx, Fxx, axes=1)
                Quu += np.tensordot(Vx, Fuu, axes=1)
                Qux += np.tensordot(Vx, Fux, axes=1)

            
            # try:
            # lam, _ = np.linalg.eig(Quu)
            # except:
            #     breakpoint()
            # mu = 1e-6
            # while np.min(lam) < 1e-3:
            #     Quu = Quu + mu*np.eye(Quu.shape[0])
            #     mu *= 2
            #     lam, _ = np.linalg.eig(Quu)

            

            # lam, _ = np.linalg.eig(Quu)
            # if np.min(lam) < 0:
            #     Quu[:,:] = Quu - 2*np.min(lam)*np.eye(Quu.shape[0])

            Quu_inv = np.linalg.inv(Quu)

            k[i] = -Quu_inv@Qu
            # if i == 0:
            #     breakpoint()

            # breakpoint()
            K[i] = -Quu_inv@Qux

            Vx = Qx - Qux.T@Quu_inv@Qu
            Vxx = Qxx - Qux.T@Quu_inv@Qux

        
        return k, K
        


        
    