
from dynamics import Dynamics

import numpy as np
import sympy as sym
from PIL import Image
from matplotlib.patches import FancyBboxPatch
import os

import matplotlib.pyplot as plt



class CartPole(Dynamics):

    def __init__(
            self,
            mp,
            mc,
            l,
            g = 9.80665,
            dt = 0.01):
        
        self.mp = mp
        self.mc = mc
        self.l = l
        self.g = g
        self.dt = dt

    def F(self, X, U, constrain = True):
        x, xdot, sin_theta, cos_theta, thetadot = X
        u = sym.tanh(U[0]) if constrain else U[0]

        mp, mc, l, g, dt = self.mp, self.mc, self.l, self.g, self.dt

        theta = sym.atan2(sin_theta, cos_theta)

        # Angular acceleration of the pendulum
        alpha = (g*sin_theta + cos_theta*(
                (-u - mp*l*thetadot**2*sin_theta)/(mc + mp)))/(
                l*(4/3 - (mp*cos_theta**2)/(mc + mp)))

        # Translational acceleration
        a = (u + mp*l*(thetadot**2*sin_theta - alpha*cos_theta))/(mc + mp)

        F = sym.Matrix([x + xdot*dt, 
                        xdot + a*dt, 
                        sym.sin(theta + thetadot*dt), 
                        sym.cos(theta + thetadot*dt), 
                        thetadot + alpha*dt])
        return F

    def plot_states(self, x, u, dt):

        sin_theta = x[:,2]
        cos_theta = x[:,3]
        theta = np.unwrap(np.arctan2(sin_theta, cos_theta))



        thetadot = x[:,4]


        fig, ax = plt.subplots(3, 1, figsize=(4, 8))
        N = u.shape[0]
        t = np.linspace(0, dt*N, N)
        ax[0].plot(theta, thetadot)
        ax[0].set_xlabel(r"$\theta (rad)$")
        ax[0].set_ylabel(r"$\dot{\theta} (rad/s)$")
        ax[0].set_title("Phase Plot")
        ax[1].set_title("Control")
        ax[1].plot(t, np.tanh(u))
        ax[1].set_xlabel("Time (s)")
        fig.tight_layout()


        fig.show()

        pass

    def animate(self, X, U, dt):
        x = X[:, 0]
        sin_theta = X[:, 2]
        cos_theta = X[:, 3]
        theta = -np.unwrap(np.arctan2(sin_theta, cos_theta)) + np.pi / 2

        u = np.tanh(U)  # Control values (scaled)
        l = self.l

        N = U.shape[0]
        t = np.linspace(0, dt * N, N)

        # Color definitions
        rich_black = "#0D1117"
        neon_red = "#F8333C"
        neon_orange = "#FEC620"
        neon_blue = "#49E9E6"
        white = "#FBFBFB"

        # Set up the plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.patch.set_alpha(0)  # Transparent background
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # Add a white border around the plot
        fig.patch.set_facecolor(white)
        ax.set_title('Cart Pole Animation', fontsize = 16, color = white)
        ax.set_xlim(1.2 * (min(x) - l), 1.2 * (max(x) + l))
        ax.set_ylim(-1.2 * l, 1.2 * l)
        ax.set_aspect('equal')
        ax.set_facecolor((1, 1, 1, 0))  # Transparent axes

        # Remove ticks and spines for a clean look
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Cart dimensions
        cart_width = 0.2 * l
        cart_height = 0.1 * l

        # Use FancyBboxPatch for rounded cart edges
        cart = ax.add_patch(FancyBboxPatch((x[0] - cart_width / 2, -cart_height / 2),
                                        cart_width, cart_height,
                                        boxstyle="round,pad=0.05",
                                        color=neon_red, ec=white, lw=2))

        # Create the pole with a white border
        pole_border, = ax.plot([x[0], x[0] + l * np.cos(theta[0])],
                            [0, l * np.sin(theta[0])],
                            color=white, linewidth=5)  # White border (thicker)
        pole, = ax.plot([x[0], x[0] + l * np.cos(theta[0])],
                        [0, l * np.sin(theta[0])],
                        color=neon_blue, linewidth=3)  # Main line (thinner)

        # Create the pole endpoint with a white border
        pole_end_border, = ax.plot(x[0] + l * np.cos(theta[0]),
                                l * np.sin(theta[0]),
                                marker='o', color=white, markersize=12)
        pole_end, = ax.plot(x[0] + l * np.cos(theta[0]),
                            l * np.sin(theta[0]),
                            marker='o', color=neon_orange, markersize=8)

       

        frame_files = []

        os.makedirs('./temp', exist_ok=True)

        def update(i):
            # Update cart position
            cart.set_bounds(x[i] - cart_width / 2, -cart_height / 2, cart_width, cart_height)

            # Update pole position
            pole_border.set_data([x[i], x[i] + l * np.cos(theta[i])], [0, l * np.sin(theta[i])])
            pole.set_data([x[i], x[i] + l * np.cos(theta[i])], [0, l * np.sin(theta[i])])

            # Update pole endpoint position
            pole_end_border.set_data([x[i] + l * np.cos(theta[i])], [l * np.sin(theta[i])])
            pole_end.set_data([x[i] + l * np.cos(theta[i])], [l * np.sin(theta[i])])

         

            # Save the frame with a transparent background
            frame_filename = f"./temp/frame_{i}.png"
            plt.savefig(frame_filename, bbox_inches='tight', transparent=True, facecolor='none', dpi=200)
            frame_files.append(frame_filename)

        # Generate and save each frame
        for i in range(N):
            update(i)

        # Create the GIF
        images = [Image.open(filename) for filename in frame_files]
        images[0].save('cartpole.gif',
                    save_all=True,
                    format='GIF',
                    append_images=images[1:],
                    duration=dt * 1000, disposal=2,
                    loop=0)

        # Clean up temporary files
        for filename in frame_files:
            os.remove(filename)
        os.rmdir('./temp')


    
    def __str__(self):
        HEADER = "\033[38;5;214m"  # Orange for the header
        BORDER = "\033[37m"        # Light gray for border and separator
        TITLE = "\033[95m"         # Pink for section titles
        VALUE = "\033[97m"         # White for values
        RESET = "\033[0m"          # Reset to default color
        
        header_art = (
            f"{BORDER}"
            f"╔═══════════════════════════════════════════════╗\n"
            f"║            {HEADER}CartPole Dynamics Model{BORDER}            ║\n"
            f"╚═══════════════════════════════════════════════╝\n"
            f"{RESET}"
        )

        symecs = (
            f"{BORDER}║{RESET} {TITLE}Mass Pole:{RESET}               {VALUE}{self.mp:<14.3f} kg   {RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Mass Cart:{RESET}               {VALUE}{self.mc:<14.3f} kg   {RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Pendulum Length:{RESET}         {VALUE}{self.l:<14.3f} m    {RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Gravity:{RESET}                 {VALUE}{self.g:<14.3f} m/s² {RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Time Step:{RESET}               {VALUE}{self.dt:<14.3f} s    {RESET} {BORDER}║{RESET}\n"
        )

        separator_line = f"{BORDER}─────────────────────────────────────────────────{RESET}\n"

        return header_art + symecs + separator_line
    
    def __repr__(self):
        HEADER = "\033[38;5;214m"  # Orange for the header and border
        TITLE = "\033[95m"         # Pink for section titles
        VALUE = "\033[97m"         # White for values
        RESET = "\033[0m"          # Reset to default color

        return (
            f"{HEADER}Quadrotor{RESET}( "
            f"{TITLE}mp{RESET}={VALUE}{self.mp:.2f} kg{RESET}, "
            f"{TITLE}mc{RESET}={VALUE}{self.mc:.2f} kg{RESET}, "
            f"{TITLE}l{RESET}={VALUE}{self.l:.2f} m{RESET}, "
            f"{TITLE}g{RESET}={VALUE}{self.g:.2f} m/s² {RESET})"
            f"{TITLE}dt{RESET}={VALUE}{self.dt:.2f} s {RESET})"
        )




