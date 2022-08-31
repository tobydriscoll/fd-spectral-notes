# Elliptic PDE

The high-dimensional version of a TPBVP is an **elliptic PDE**. These typically represent a system in steady state. The most famous is **Poisson's equation**, which in 2D is 

$$
\Delta u = \partial_{xx}u + \partial_{yy}u = f(x,y), \quad (x,y) \in \Omega \subset \mathbb{R}^2. 
$$

This equation represents the steady state of a diffusive process. The function $f$ might be called the **loading function** in some contexts. If $f(x,y)\equiv 0$, we get **Laplace's equation**. 

For Poisson's equation we need a boundary condition imposed on the entire boundary $\partial \Omega$. It may be Dirichlet, which prescribes the value of $u$, Neumann, which prescribes $\pp{u}{n}$, or a mixture, and the type can change along the boundary.

