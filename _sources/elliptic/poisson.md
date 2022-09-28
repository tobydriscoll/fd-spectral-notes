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

# Solving the Poisson equation

$$
    \mathbf{A}\otimes \mathbf{B} =
    \begin{bmatrix}
    A_{11} \mathbf{B} & A_{12}\mathbf{B} & \cdots & A_{1n}\mathbf{B} \\
    A_{21} \mathbf{B} & A_{22}\mathbf{B} & \cdots & A_{2n}\mathbf{B} \\
    \vdots & \vdots &  & \vdots \\
    A_{m1} \mathbf{B} & A_{m2}\mathbf{B} & \cdots & A_{mn}\mathbf{B}
    \end{bmatrix}.
$$

+++

$$
\underbrace{\bigl[ ({\mathbf{I}_{y}} \otimes {\mathbf{D}_{xx}}) + ({\mathbf{D}_{yy}}\otimes {\mathbf{I}_{x}})\bigr]}_{\mathbf{A}} \, \mathbf{u} = \mathbf{f},
$$

```{code-cell}
f = (x,y) -> x^2 - y + 2;
using LinearAlgebra
⊗ = kron
```

```{code-cell}
foreach(println,readlines("/Users/driscoll/817/notes/elliptic/diffmats.jl"))
```

```{code-cell}
include("/Users/driscoll/817/notes/elliptic/diffmats.jl")
m,n = 7,5
x,Dx,Dxx = diffmats(m,0,3)
y,Dy,Dyy = diffmats(n,-1,1)
unvec = u -> reshape(u,m+1,n+1);
```

```{code-cell}
F = [ f(x,y) for x in x, y in y ]
```

```{code-cell}
A = I(n+1)⊗Dxx + Dyy⊗I(m+1)
b = vec(F);
```

```{code-cell}
@show N = length(F);
```

```{code-cell}
A
```

```{code-cell}
using Plots
default(size=(400,400))
spy(A,color=:blues,m=3,
    title="System matrix before boundary conditions")
```

```{code-cell}
isboundary = trues(m+1,n+1)
isboundary[2:m,2:n] .= false
idx = vec(isboundary);
```

```{code-cell}
spy(isboundary,m=3,color=:darkblue,legend=:none,
    title="Boundary points",
    xaxis=("column index",[0,n+2]),yaxis=("row index",[0,m+2]) )
```

```{code-cell}
I_N = I(N)
A[idx,:] .= I_N[idx,:];     # Dirichlet conditions
```

```{code-cell}
spy(sparse(A),color=:blues,m=3,
    title="System matrix with boundary conditions")    
```

```{code-cell}
b[idx] .= 1;                 # Dirichlet values
```

Now we can solve for $\mathbf{u}$ and reinterpret it as the matrix-shaped $\mathbf{U}$, the solution on our grid.

```{code-cell}
u = A\b
U = unvec(u)
```

### Implementation

```{code-cell}
include("/Users/driscoll/817/notes/elliptic/poisson.jl")
foreach(println,readlines("/Users/driscoll/817/notes/elliptic/poisson.jl"))
```

We can engineer an example by choosing the solution first. Let $u(x,y)=\sin(3xy-4y)$. Then one can derive $f=\Delta u = -\sin(3xy-4y)\bigl(9y^2+(3x-4)^2\bigr)$ for the forcing function and use $g=u$ on the boundary.

First we define the problem on $[0,1]\times[0,2]$.

```{code-cell}
f = x -> -sin(3x[1]*x[2]-4x[2]) * (9x[2]^2+(3x[1]-4)^2)
g = x -> sin(3x[1]*x[2]-4x[2])
xspan = [0,1];  yspan = [0,2];
```

Here is the finite-difference solution.

```{code-cell}
x,y,U = poissonfd(f,g,80,xspan,100,yspan);
```

```{code-cell}
surface(x,y,U',color=:viridis,
    title="Solution of Poisson's equation",      
    xaxis=("x"),yaxis=("y"),zaxis=("u(x,y)"),
    right_margin=3Plots.mm,camera=(70,50))    
```

The error is a smooth function of $x$ and $y$. It must be zero on the boundary; otherwise, we have implemented boundary conditions incorrectly.

```{code-cell}
error = [g([x,y]) for x in x, y in y] - U;
M = maximum(abs,error)
contour(x,y,error',levels=17,aspect_ratio=1,
    clims=(-M,M),color=:redsblues,colorbar=:bottom,
    title="Error",xaxis=("x"),yaxis=("y"),
    right_margin=7Plots.mm)
plot!([0,1,1,0,0],[0,0,2,2,0],l=(2,:black))
```

## Neumann conditions

If we use the boundary row replacement method, generalizing the above to Neumann conditions can be surprisingly easy. The key steps for the Dirichlet case were to define a vector `idx` indicating which rows correspond to boundary nodes, and then make replacements:

:::{code-block} julia
A[idx,:] .= I(N)[idx,:]    # Dirichlet conditions
b[idx] .= g.(grid[idx])     # Dirichlet values
:::

(We're ignoring the row scaling step for simplicity.) 

For a Neumann condition, all we have to do is swap the identity operator (matrix) for one that computes the outward normal derivatives. In the homogeneous case, we don't even need to be concerned with the distinction between inward and outward normals. The only new complication is that we need different operators for the boundaries in the $x$ and $y$ directions. For example, to impose homogeneous Neumann conditions along both edges with constant $x$ values, we use

:::{code-block} julia
xboundary = trues(m+1,n+1)
xboundary[2:m,:] .= false
idx = vec(xboundary)
A[idx,:] .= kron(I(n+1),Dx)[idx,:]    # Neumann conditions
b[idx] .= 0                           # Neumann values
:::

## Sparsity

The implementation of `diffmats` changed just a bit from the 1D case:

```{code-cell}
foreach(println,readlines("diffmats.jl"))
```

The differentiation matrices are returned in **sparse** form. While they are sparse even considered as 1D matrices, the sizes were too small to worry about it then. But now we should be more careful. The system matrix is of size $O(mn)$, while the number of nonzero elements per row is small and bounded by constant. 

Say $m=O(n)$ for simplicity. Then a naive dense factorization of the system would take $O(n^6)$ operations. Exploiting sparsity, the count should be no more than $O(n^4)$, which still grows quickly but is far better.

## Nonlinear problems

A nonlinear elliptic PDE, like a nonlinear TPBVP, leads to a nonlinear algebraic system. As before, we usually want to apply a quasi-Newton method to solve that system. This will entail the solution of multiple linear systems with a changing matrix (Jacobian or approximation). 

Unless $m$ and $n$ are quite small, the linear solver within the nonlinear iterations will have to support sparsity. Some solvers accept a sparsity pattern that they can exploit; in most cases you are best off supplying your own Jacobian.

For example, to solve 

$$
\epsilon \Delta u + u u_x + 1 = 0, 
$$

with Dirichlet boundary value 1, note that the linearization of the PDE around function $u_0$ is 

$$
\epsilon \Delta + u_0 \partial_x + (\partial_x u_0),
$$

and it has homogeneous Dirichlet BC.

```{code-cell}
include("/Users/driscoll/817/notes/elliptic/diffmats.jl")
n = 60
x,Dx,Dxx = diffmats(n,0,1)
y,Dy,Dyy = diffmats(n,0,1)
bdy = trues(n+1,n+1)
bdy[2:n,2:n] .= false
bdy = vec(bdy)

function residual(u) 
    U = reshape(u,n+1,n+1)
    R = 0.05*(Dxx*U + U*Dyy') + U.*(Dx*U) .+ 1
    R[bdy] .= U[bdy] .- 1
    return vec(R)
end;

function jac(u)
    L = kron(I(n+1),Dxx) + kron(Dyy,I(n+1))
    U_x = Dx*reshape(u,n+1,n+1)
    J = 0.05L + spdiagm(u)*kron(I(n+1),Dx) + spdiagm(vec(U_x))
    J[bdy,:] .= I((n+1)*(n+1))[bdy,:]
    return J
end;
```

```{code-cell}
using NLsolve
sol = nlsolve(residual,jac,vec(zeros(n+1,n+1)));
sol.residual_norm
```

```{code-cell}
sol
```

```{code-cell}
contour(x,y,reshape(sol.zero,n+1,n+1)',aspect_ratio=1)
```

```{code-cell}

```
