# Finite differences and spectral methods for PDEs

Here are the three major PDE problem types:

```{list-table} PDE Problems
:header-rows: 1
:name: table-pdes

* - 
  - Nonlinear
  - Linear
* - Boundary-value
  - $F(u)=0$
  - $Lu = 0$
* - Time-dependent
  - $u_t = F(u)$
  - $u_t = Lu$
* - Eigenvalue
  - $-$
  - $Au = \lambda Bu$
```

In each case, $u$ is a (possibly vector-valued) function of space, and perhaps time, and the operators include differential operations in space dimensions. Boundary-value problems represent steady state and require boundary conditions depending on the order of the differential operator. (Of course, the underlying problem is an ODE, not a PDE, if there is just one space dimension.) Time-dependent problems also require an initial condition; we have limited ourselves to first-order in time, but there are ways to transform higher-order problems down to first-order by adding dimensions to $u$. Finally, while there are ways to pose nonlinear eigenvalue problems, the linear case is the standard one and has many applications in physics.

Since the solution of a PDE is a function, our first decision in a numerical solution is how to represent the function on a computer. The **finite-difference** approach is to replace the function with a collection of its values at specific locations called **nodes** in space and/or time. Typically, one then determines those values through approximations to the differential operators in the original problem. When the approximations are imposed pointwise at the same set of nodes as the function values, the resulting approach may be called **collocation.**

A finite-difference method does not directly address the connection between the collocated values and an approximation to the solution function $u(x)$. One might use simple piecewise linear interpolation, for example, if this function is desired, but there is no intrinsic need to do so. By contrast, a **spectral method** assumes that the discrete function values are connected to an interpolating or least-squares approximation of maximum accuracy for smooth functions. Alternatively, the approximate solution may be viewed as a linear combination of basis functions with optimal approximation properties. (Sometimes, in order to distinguish from a spectral method based on expansion coefficients, a spectral collocation method is referred to as a *pseudospectral* method, although this term is not universally liked.) These methods can be viewed through the lens of finite differences or through the lens of classical orthogonal polynomials and Fourier transformations. 

The point of finite differences or spectral collocation is to reduce the original continuous problem to a discrete system: linear or nonlinear algebraic equations, ODEs, or a matrix eigenvalue problem. The finite-dimensional analogs are all major problem types in scientific computation; in fact, the interest in and development of the finite-dimensional problems is driven to a great extent by discretizations of PDEs. We might then ask whether the discrete solution **converges** to the continuous one as the size of the discrete problem increases. There are two elements that create convergence: **accuracy**, which is how well the discrete problem represents the original one, and **stability**, whose definition is dependent on the problem type but essentially ensures that the discrete mapping from data to solution does not grow without bound as the problem size increases to infinity.