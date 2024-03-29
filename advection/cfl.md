---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Julia 1.8.0
  language: julia
  name: julia-1.8
---

# The CFL condition

Let's start with the linear 1D advection equation

$$
\partial_t u + c \partial_x u = 0.
$$

For any periodic differentiable function $\phi$, a solution is 

$$
u(x,t) = \phi(x-ct), 
$$

which represents transport at velocity $c$. If an initial condition is supplied, then $\phi(x)=u_0(x)$. 

There is a necessary condition on solvers that can be deduced from global behaviors without getting far into the details of a discretization. 

## Domains of dependence

::::{prf:definition} Domain of dependence
Let $u(x,t)$ be the solution of an evolutionary PDE with initial condition $u_0(x)$. The  **domain of dependence** of the solution at $(x,t)$ is the set of all $x$ such that $u_0(x)$ can possibly affect $u(x,t)$. 
::::

In the advection equation, the domain of dependence at $(x,t)$ is the single point $\{x-ct\}$, and the upwind direction is to the left or to the right of $x$ if $c$ is positive or negative, respectively.

::::{prf:definition} Numerical domain of dependence
Let $U_{i,j}$ be the approximate solution of an evolutionary PDE at $x=x_i$, $t=t_j$ from a numerical method, when the initial condition is given by $U_{i,0}$ for all $i$. The **numerical domain of dependence** of the method at $(x_i,t_j)$ is the set of all $x_i$ such that $U_{i,0}$ can possibly affect $U_{i,j}$.
::::

::::{prf:example}
Suppose we discretize by a centered difference in space and Euler in time. The numerical solution $U_{i,j}$ at $x=x_i$, $t=t_j$ can depend directly only on $U_{i-1,j}$, $U_{i,j}$, and $U_{i+1,j}$. Going back another time step, the dependence extends to space positions $i-2$ and $i+2$, and so on. When we reach the initial time, the dependence of $U_{i,j}$ reaches from $x_{i-j}$ to $x_{i+j}$, or between $x_i-jh$ and $x_i+jh$. 

For any particular discretization this set is discrete, but if the step sizes are taken to zero in a fixed ratio, the numerical DoD fills in the region between the extremes. In the absence of boundaries, the situation is illustrated in {numref}`figure-cflpicture`.
::::

```{figure} cflpicture.svg
:name: figure-cflpicture
:width: 450px
Numerical domain of dependence for the centered + Euler scheme. If $\tau$ and $h$ are infinitesimally small, the shaded region is filled in.
```

## The criterion

Here is a celebrated theorem that also provides a handy rule of thumb.

::::{prf:theorem} Courant–Friedrichs–Lewy (CFL) condition
In order for a numerical method for an advection equation to converge to the correct solution, the limiting numerical domain of dependence must contain the exact domain of dependence.
::::


::::{prf:example}
Returning to the previous example, the numerical domain of dependence depicted in {numref}`Figure {number} <figure-cflpicture>` contains the exact domain of dependence $\{x_i-c t_j\}$ only if $x_i-j h \le x_i -c t_j \le x_i+jh$, or $|c j\tau|\le j h$. That is,

:::{math}
  \frac{h}{\tau} \ge |c|, \quad  \tau,h\rightarrow 0.
:::

We can view this condition as a restriction on the time step: $\tau \le h/|c|$. 
::::

:::{caution} 
The CFL condition is a *necessary* criterion for convergence, but not a *sufficient* one. For instance, we could define $U_{i,j}$ to be the average value of the initial condition for all $j>0$. While that would make the numerical domain of dependence equal to the entire domain, this method has nothing to do with solving a PDE correctly!
:::

Note that an implicit IVP integrator leads to automatic satisfaction of the CFL condition, because one time step induces (possible) dependence on all the previous values. 

One way to interpret the restriction from the above example transfers nicely to other problems as a rule of thumb: *the numerical method must allow propagation at at least the maximum speed possible in the solution*. For instance, if you want to simulate weather using an explicit method on a grid with cell size 1 km while tracking wind speeds up to 200 km/hr, then you will need a time step no larger than 1/200 hr in order to even have a chance at stability.

## Upwind and downwind

The CFL condition is about the *shape* of the domain of dependence as well as its size. 

::::{prf:example}
Suppose that $c>0$ and we pair a backward difference in space with an Euler solver in time. The solution value $U_{i,j}$ depends on $U_{i,j-1}$ and $U_{i-1,j-1}$. Each step we go backward in time extends us another grid point to the left. If $\tau/h$ is held constant as both go to zero, then at $t=0$ the numerical domain of dependence at $(x_i,t_j)$ is $[x_i-jh,x_i]$. The condition that the exact DoD $x_i-c(j\tau)$ lie in this interval requires that

$$
x_i-jh \le x_i - cj\tau \qquad \Rightarrow \qquad \tau \ge h/c,
$$

which is what we found for the centered difference in space. 

If, however, we use a forward difference in space, the numerical DoD is the interval $[x_i,x_i+jh]$, and it is impossible for the exact DoD to lie within it. The CFL condition cannot be satisfied.
::::

If the exact domain of dependence at $(x,t)$ lies entirely in one direction relative to $x$, then that direction is called the **upwind** direction of the PDE, and its opposite is the **downwind** direction. As the example above illustrates, spatial discretizations must have a DoD that extends into the upwind direction. 

There's a simple intuitive interpretation. For $c>0$ as above, the transport in the PDE is from left to right. Sitting at one node, the proper way to forecast the future at that node is to look in the upwind direction. The future solution does not depend at all on what lies to the right, and building your finite difference looking only in that direction cannot lead to convergence. For $c<0$, the roles of forward and backward differences would be reversed: the upwind direction requires the forward difference. 

Earlier we saw that a centered difference is useful regardless of the sign of $c$. So why not always use centered differences? There are multiple answers, but maybe the most robust one is that in practice we are often interested in problems with steep gradients or *shocks*, and differencing across that gradient is detrimental to accuracy (or potentially just wrong, for actual shocks).
