# Advection

The archetypical linear model of advection in one dimension is the PDE

$$
\partial_t u + c \partial_x u = 0,
$$

where the constant $c$ is the velocity of travel for any initial condition. This is a **hyperbolic PDE**. It requires one initial condition and one boundary condition; the BC must represent an inflow condition, as we will see.

Another important linear hyperbolic PDE is the **wave equation**, which in 1D is 

$$
\partial_{tt} u + c^2 \partial_{xx}. 
$$

The wave equation allows solutions of velocities $c$ and $-c$ simultaneously. It requires two boundary conditions and initial conditions on both $u$ and $\partial_t u$, unless it is reformulated as **Maxwell's equations**,

$$
\partial_t v &= \partial_x w, \\ 
\partial_t w &= -\partial_x v. 
$$
