
from dynamics import Dynamics

import sympy as sym

"""
These dynamics model a quadrotor in a "X" configuration. The body coordinate frame is aligned such that 
the X-direction is direction halfway between the first and second rotor, the Y-direction is halfway between 
the first and fourth rotor (if the X-direction is forward, the Y-direction is left), and the Z-direction up 
through the body (as is consistent with the right hand rule). There is no drag or motor dynamics considered 
in this model. Inertial coordinates are in NWU (north, west, up). The transformation from the inertial 
coordinates to the body coodrinates is defined by an Euler 1-2-3 rotation R(ϕ, θ, ψ)^T = R1(ϕ)R2(θ)R3(ψ)
"""

class Quadrotor(Dynamics):

    def __init__(
            self,
            mass,
            Ixx,
            Iyy,
            Izz,
            kt,
            l,
            g = 9.80665,
            dt = 0.01):
        self.mass = mass # Quadrotor mass (kg)

        self.Ixx = Ixx # Quadrotor moment of inertia about x-axis (kg·m²)
        self.Iyy = Iyy # Quadrotor moment of inertia about y-axis (kg·m²)
        self.Izz = Izz # Quadrotor moment of inertia about z-axis (kg·m²)

        self.l = l
        self.kt = kt # Rotor torque constant (1/m)
        self.g = g # Gravitational constant (m/s²)
        self.dt = dt # Time step (s)

        self.states, self.controls, self.rates = self.build_symbolic_dynamics()

    def build_symbolic_dynamics(self):
        # Define state variables
        x, y, z = sym.symbols('x y z') # Position components states
        xdot, ydot, zdot = sym.symbols('xdot ydot zdot')  # Velocity components states
        phi, theta, psi = sym.symbols('phi theta psi')  # Euler rotation angles states
        p, q, r = sym.symbols('p q r')  # Rotation rates states
        
        # Define control inputs
        u1, u2, u3, u4 = sym.symbols('u1 u2 u3 u4')
        
        # Rotation matrix R (using Euler angles)
        R = sym.Matrix([
            [                                      sym.cos(theta)*sym.cos(psi),                                          sym.cos(theta)*sym.sin(psi),               -sym.sin(theta)],
            [sym.sin(phi)*sym.sin(theta)*sym.cos(psi) - sym.cos(phi)*sym.sin(psi),    sym.sin(phi)*sym.sin(theta)*sym.sin(psi) + sym.cos(phi)*sym.cos(psi),    sym.sin(phi)*sym.cos(theta)],
            [sym.cos(phi)*sym.sin(theta)*sym.cos(psi) + sym.sin(phi)*sym.sin(psi),    sym.cos(phi)*sym.sin(theta)*sym.sin(psi) - sym.sin(phi)*sym.sin(psi),    sym.cos(phi)*sym.cos(theta)]
                       ])
        
        # Translational acceleration
        a = 1/self.mass*(R*sym.Matrix([[0], [0], [u1 + u2 + u3 + u4]]) + sym.Matrix([[0], [0], [-self.mass*self.g]]))

        # Angular velocity in the inertial reference frame (This is where the gimbal lock singularity appears)
        omegadot = sym.Matrix([
            [1,    sym.sin(phi)*sym.tan(theta),    sym.cos(phi)*sym.tan(theta)],
            [0,                  sym.cos(phi),                 -sym.sin(phi)],
            [0,    sym.sin(phi)/sym.cos(theta),    sym.cos(phi)/sym.cos(theta)]
        ])*sym.Matrix([[p], [q], [r]])

        # Angular acceleration in the body reference frame
        alpha = sym.Matrix([[(sym.sqrt(2)/2*(u1 + u3 - u2 - u4)*self.l - (self.Izz - self.Iyy)*q*r)/self.Ixx],
                           [(sym.sqrt(2)/2*(u3 + u4 - u1 - u2)*self.l - (self.Izz - self.Ixx)*p*r)/self.Iyy],
                           [                                        (self.kt*(u1 + u4 - u2 - u3))/self.Izz]
                          ])

        # Consolidate dynamics
        states = sym.Matrix([[x], [y], [z], [xdot], [ydot], [zdot], [phi], [theta], [psi], [p], [q], [r]])
        controls = sym.Matrix([[u1], [u2], [u3], [u4]])
        rates = sym.Matrix.vstack(
            sym.Matrix([xdot, ydot, zdot]),
            a,
            omegadot,
            alpha,
        )

        return states, controls, rates

    def get_symbolic_states(self):
        return self.states

    def get_symbolic_controls(self):
        return self.controls

    def get_symbolic_rates(self):
        return self.rates
    
    def get_symbolic_discrete_dynamics(self):
        return self.get_symbolic_states() + self.rates*self.dt
    
    def __str__(self):
        HEADER = "\033[38;5;214m"  # Orange for the header
        BORDER = "\033[37m"        # Light gray for border and separator
        TITLE = "\033[95m"         # Pink for section titles
        VALUE = "\033[97m"         # White for values
        RESET = "\033[0m"          # Reset to default color
        
        header_art = (
            f"{BORDER}"
            f"╔═══════════════════════════════════════════════╗\n"
            f"║            {HEADER}QUADROTOR DYNAMICS MODEL{BORDER}           ║\n"
            f"╚═══════════════════════════════════════════════╝\n"
            f"{RESET}"
        )

        symecs = (
            f"{BORDER}║{RESET} {TITLE}Mass:{RESET}                    {VALUE}{self.mass:<14.3f} kg   {RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Inertia:{RESET}                 {VALUE}Ixx={self.Ixx:<10.3f} kg·m²{RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {'':<25}{VALUE}Iyy={self.Iyy:<10.3f} kg·m²{RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {'':<25}{VALUE}Izz={self.Izz:<10.3f} kg·m²{RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Rotor Torque Const:{RESET}      {VALUE}{self.kt:<14.3f} 1/m  {RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Arm Length:{RESET}              {VALUE}{self.l:<14.3f} m    {RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Gravity:{RESET}                 {VALUE}{self.g:<14.3f} m/s² {RESET} {BORDER}║{RESET}\n"
            f"{BORDER}║{RESET} {TITLE}Time Step:{RESET}               {VALUE}{self.dt:<14.3f} s {RESET} {BORDER}   ║{RESET}\n"
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
            f"{TITLE}mass{RESET}={VALUE}{self.mass:.2f} kg{RESET}, "
            f"{TITLE}Ixx{RESET}={VALUE}{self.Ixx:.3f} kg·m²{RESET}, "
            f"{TITLE}Iyy{RESET}={VALUE}{self.Iyy:.3f} kg·m²{RESET}, "
            f"{TITLE}Izz{RESET}={VALUE}{self.Izz:.3f} kg·m²{RESET}, "
            f"{TITLE}kt{RESET}={VALUE}{self.kt:.3f} 1/m{RESET}, "
            f"{TITLE}l{RESET}={VALUE}{self.l:.2f} m{RESET}, "
            f"{TITLE}g{RESET}={VALUE}{self.g:.2f} m/s² {RESET})"
            f"{TITLE}dt{RESET}={VALUE}{self.dt:.2f} s {RESET})"
        )

