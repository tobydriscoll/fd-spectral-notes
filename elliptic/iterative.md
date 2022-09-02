---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Julia 1.8.0
  language: julia
  name: julia-1.8
---

# Iterative linear algebra

Even exploiting sparsity, the time to solve a linear system for an elliptic PDE quickly becomes unreasonable as the grid size increases. In terms of the grid spacing $h$, the number of unknowns scales as $O(h^{-2})$ and the time per linear system solution is $O(h^{-4})$. With a convergence rate of 2nd order, a 10-fold decrease in the error requires $h$ to decrease by a factor of $3.3$ and the execution time goes up by a factor of about 120. The situation is even more unmanageable in 3D, where the solution time scales with $O(h^{-6})$. 

There's something unsatisfying about solving a linear system exactly (modulo roundoff) when the solution is itself a low-accuracy representation of the continuous object that is the real target. In other words, if a nodal value has only 3 accurate digits, we don't care that the solution of the discrete system is 0.123456787654, as we are just as happy with 0.123. But factorization-based linear solvers don't give us a choice in the matter.

Thus, as the number of unknowns grows, we seek iterative methods to solve the linear systems. We can stop the iterations once we have as much accuracy as we hope to get from the discretization, potentially saving a lot of time.

## Splitting methods

The OG iterative methods for linear systems are known as **splitting methods**. We rewrite the system matrix as $\bfA=\mathbf{M}-\mathbf{N}$, and $\bfA\bfu=\bfb$ as 

$$
\mathbf{M} \bfu = \mathbf{N} \bfu + \bfb. 
$$

We then perform a fixed-point iteration

$$
\mathbf{M} \bfu_{k+1} = \mathbf{N} \bfu_k + \bfb. 
$$

Clearly this is designed to use a matrix $\mathbf{M}$ that makes it easy to solve the FP systems quickly. The most famous methods are **Jacobi**, which uses

$$
\mathbf{M} = \diag(\diag(\bfA)), 
$$

and **Gauss--Seidel**, which lets $\mathbf{M}$ be the lower triangle of $\bfA$. The convergence of the method depends on convergence of powers of $\mathbf{M}^{-1}\mathbf{N}$ to zero. For a few cases, including Poisson's equation, this convergence is well-understood, to an extent that allows a better variant called *SOR*.

The only reasons to use splitting methods are (a) for a *multigrid* method, which uses the convergence properties cleverly, and/or (b) massively parallel implementations of Jacobi, for which the components of $\bfu_{k+1}$ may be computed independently.

## Krylov methods

In most contexts the preferred iterative methods are **Krylov subspace iterations**. For particular problem types, the best-known choices are

* Any matrix: GMRES
* Symmetric matrix: MINRES 
* SPD matrix: Conjugate gradients (though MINRES is fine)

There are many others, such as QMR, Bi-CGStab, etc.

All of the methods look for a solution of $\bfA\bfu=\bfb$ in the nested subspaces spanned by $\bfb,\bfA\bfb,\bfA^2\bfb,\dots$. They use different iterations to generate them and to find the optimal solution for different definitions of optimality. 

One extremely useful aspect of these methods is that the only way they use the matrix $\bfA$ is by computing the matrix product $\bfA\bfv$ for an arbitrary vector $\bfv$. That is, they just apply the linear transformation implied by $\bfA$. This enables **matrix-free iterations** in which $\bfA$ is never explicitly assembled.

+++

Let's revisit the Poisson equation with $f=\Delta u = -\sin(3xy-4y)\bigl(9y^2+(3x-4)^2\bigr)$ for the forcing function and $g=\sin(3xy-4y)$ on the boundary, which has $g$ as the solution everywhere.

```{code-cell}
f = x -> -sin(3x[1]*x[2]-4x[2]) * (9x[2]^2+(3x[1]-4)^2)
g = x -> sin(3x[1]*x[2]-4x[2])
xspan = [0,1];  yspan = [0,2];
```

Most of the problem setup is the same as for the direct system method.

```{code-cell}
m,n = 40,60 
include("diffmats.jl")
x,Dx,Dxx = diffmats(m,xspan...)
y,Dy,Dyy = diffmats(n,yspan...)
grid = [(x,y) for x in x, y in y]

# Identify boundary locations.
isboundary = trues(m+1,n+1)
isboundary[2:m,2:n] .= false
idx = vec(isboundary);

# forcing function / boundary values vector
b = vec( f.(grid) )
b[idx] = g.(grid[idx]);   # assigned values
```

Now, instead of using Kronecker products (followed by boundary modifications) to build the matrix $\bfA$, we define a function that applies $\bfA$ using the Sylvester form of the problem.

```{code-cell}
# Apply Laplacian operator with Dirichlet condition.
function laplacian(v)
    V = reshape(v,m+1,n+1)
    AV = Dxx*V + V*Dyy'
    AV[idx] .= V[idx]   # Dirichlet condition
    return vec(AV)
end

using Krylov,LinearMaps
A = LinearMap(laplacian,(m+1)*(n+1))
```

```{code-cell}
u,stats = gmres(A,b,rtol=1e-6)
stats
```

```{code-cell} julia
using Plots
# plotlyjs()
U = reshape(u,m+1,n+1)
surface(x,y,U',color=:viridis,
    title="Solution of Poisson's equation",      
    xaxis=("x"),yaxis=("y"),zaxis=("u(x,y)"))    
```

```{code-cell}
contour(x,y,(U-g.(grid))',color=:viridis,aspect_ratio=1,
    title="Error",      
    xaxis=("x"),yaxis=("y"),zaxis=("u(x,y)"),
    right_margin=3Plots.mm,camera=(70,50))   
```

Note that while the Laplacian operator is nominally symmetric and negative definite, boundary conditions can wreck that structure.

## Preconditioning 

Krylov methods converge at a rate that depends strongly on the underlying linear operator. For symmetric matrices, the dependence can be characterized in terms of the spectrum of the matrix. When the ratio between the eigenvalues farthest from and closest to the origin is very large, the convergence becomes unacceptably slow. 

In the case of Poisson's equation, a second-order FD method produces a matrix with condition number $O(h^{-2})$ when using grid spacing $h$, and the number of MINRES or CG iterations grows as $O(h^{-1})$. 

The best response is to use a **preconditioner**, which is a way to apply an approximate solution process to improve the convergence rate to the actual solution. For example, if one uses a recursive coarsening procedure to approximate the original FD method, the result is a *multigrid* preconditioner that can provide convergence in $O(1)$ iterations in ideal circumstances. Another approach is to decompose the domain into pieces and simplify or ignore the interactions between them.


## Newton--Krylov methods

In a nonlinear problem, we have an *outer iteration* of changing linear problems for the nonlinear part, and an *inner iteration* of a Krylov method to solve the linear part. Typically one starts the inner iterations with a rather large error tolerance, since finding accurate values for a bad solution to the nonlinear problem is a waste of time. This tolerance decreases as the outer iteration homes in on the solution of the nonlinear problem. Even so, the Newton corrections need be found accurately relative to the outer solution only.  