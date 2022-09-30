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

# Multiple space dimensions

For problems posed in multiple space dimensions on tensor-product domains, the techniques applied for elliptic problems carry over into the method of lines. If the discretization size is not too large, your implementation may be able to work entirely with the rectangular shape for a grid function and not use reshaping and Kronecker products. Again, you may have an option between using boundary conditions to remove degrees of freedom from the ODEs, or using a DAE formulation to impose the BCs on a full-sized discretization.

For example, consider the nonlinear advection--diffusion problem

$$
\partial_t u + u (\partial_x u + \partial_y u) = \mu \Delta u
$$

on the rectangle $[0,3] \times [-1,1]$, where $u=0$ on $y=\pm 1$ and homogeneous Neumann conditions are applied on the vertical sides of the boundary.

First, let's consider how to remove boundary values from the unknowns by implicit application of the boundary conditions. For the Neumann conditions we will use the one-sided FD formula

$$
-\tfrac{3}{2}u(0) + 2 u(h) - \tfrac{1}[2}u(2h) = 0
$$

and its antireflection at the right boundary. The equation above defines $u$ at the boundary in terms of interior values, so we don't have to include the boundary values in our ODE unknowns.

We'll use two functions to switch between the compact, vector variable and the grid values matrix.

```{code-cell}
pack(U) = vec(U[2:m,2:n]);
function unpack(u)
    U = zeros(m+1,n+1)
    U[2:m,2:n] .= reshape(u,m-1,n-1)
    # Homogeneous Dirichlet in y
    U[:,1] .= 0
    U[:,n+1] .= 0
    # Homogeneous Neumann in x
    U[1,:] .= (4U[2,:] - U[3,:])/3
    U[m+1,:] .= (4U[m,:] - U[m-1,:])/3
    return U
end;
```

```{code-cell}
function timederiv(u,μ,t)
    U = unpack(u)
    Ux,Uy = Dx*U,U*Dy'
    LU = Dxx*U + U*Dyy'
    Ut = -U.*(Ux+Uy) + μ*LU
    return pack(Ut)
end;   
```

```{code-cell}
include("diffmats.jl")
using OrdinaryDiffEq

m,n = 44,32
x,Dx,Dxx = diffmats(m,0,3)
y,Dy,Dyy = diffmats(n,-1,1)
init = x -> (2+cos(π*x[1]))*(1-x[2]^2)
U₀ = [ init([x,y]) for x in x, y in y ];
```

```{code-cell}
IVP = ODEProblem(timederiv,pack(U₀),(0.,1.),0.4)
sol = solve(IVP,Rodas4P(autodiff=false));
```

```{code-cell}
using Plots,PyFormattedStrings

anim = @animate for t in range(0,1,51)
    contour(x,y,unpack(sol(t))',levels=range(0,3,31),aspect_ratio=1,
        clims=(0,3),fill=true,title=f"t={t:.2f}",color=:viridis)
end
mp4(anim,"multidim1.mp4")
```

Here is the same problem approached as a DAE. We can use the full grid matrix as the unknowns and just designate the boundary points as algebraic variables via the mass matrix. Note that the mass matrix is constructed in sparse form.

```{code-cell}
function timederiv!(du,u,μ,t)
    U = reshape(u,m+1,n+1)
    Ux,Uy = Dx*U,U*Dy'
    LU = Dxx*U + U*Dyy'
    dU = -U.*(Ux+Uy) + μ*LU
    dU[:,1] .= U[:,1] 
    dU[:,n+1] .= U[:,n+1]
    dU[1,:] .= -(-1.5U[1,:] + 2U[2,:] - 0.5U[3,:])/(x[2]-x[1])
    dU[m+1,:] .= (1.5U[m+1,:] - 2U[m,:] + 0.5U[m-1,:])/(x[2]-x[1])
    du .= vec(dU)
    return du
end;
```

```{code-cell}
using SparseArrays
M = zeros(m+1,n+1)
M[2:m,2:n] .= 1
∂ₜ = ODEFunction(timederiv!,mass_matrix=spdiagm(vec(M)))
IVP = ODEProblem(∂ₜ,vec(U₀),(0.,1.),0.4)

sol = solve(IVP,Rodas4P(autodiff=false));
```

```{code-cell}
anim = @animate for t in range(0,1,51)
    contour(x,y,reshape(sol(t),m+1,n+1)',levels=range(0,3,31),aspect_ratio=1,
        clims=(0,3),fill=true,title=f"t={t:.2f}",color=:viridis)
end
mp4(anim,"multidim2.mp4")
```

On my machine, the DAE version finishes a lot faster.

+++

## Explicit Jacobians

We may well find that the time required to solve an IVP of this type grows quickly as a function of discretization size, particularly if the problem includes significant diffusion. Since diffusion tends to be stiff, we probably will turn to an implicit solver, which has to solve nonlinear equations at each time step. For example, applying the trapezoid formula with step size $\tau$ to a nonlinear problem $\partial_t u = f(t,u)$ must solve

$$
\mathbf{u}_{k+1} - \tfrac{1}{2}\tau \mathbf{f}(t_{k+1},\mathbf{u}_{k+1}) = \mathbf{u}_{k} + \tfrac{1}{2}\tau \mathbf{f}(t_{k+1},\mathbf{u}_{k})
$$

to get $\mathbf{u}_{k+1}$. The Jacobian of this nonlinear system is

$$
\mathbf{I} - \tfrac{1}{2}\tau \mathbf{J},
$$

where $\mathbf{J}$ is the Jacobian of just $\mathbf{f}$. Hence, we require the same information as for solving a steady problem with $\mathbff{f}$. Furthermore, we might want to use Newton--Krylov methods, for example, when solving the nonlinear systems.

Fortunately, the DAE code above seems to work properly with automatic differentiation to find the exact Jacobian, though it doesn't necessarily result in a big speedup in this example.

```{code-cell}
m,n = 42,100
x,Dx,Dxx = diffmats(m,0,3)
y,Dy,Dyy = diffmats(n,-1,1)
U₀ = [ init([x,y]) for x in x, y in y ];

M = zeros(m+1,n+1)
M[2:m,2:n] .= 1
∂ₜ = ODEFunction(timederiv!,mass_matrix=spdiagm(vec(M)))
IVP = ODEProblem(∂ₜ,vec(U₀),(0.,0.8),0.25);
```

```{code-cell}
println("using FD Jacobian:")
@elapsed sol = solve(IVP,Rodas4P(autodiff=false))
```

```{code-cell}
println("using autodiff Jacobian:")
@elapsed sol = solve(IVP,Rodas4P(autodiff=true))
```

```{code-cell}
anim = @animate for t in range(0,1,51)
    contour(x,y,reshape(sol(t),m+1,n+1)',levels=range(0,3,31),aspect_ratio=1,
        clims=(0,3),fill=true,title=f"t={t:.2f}",color=:viridis)
end
mp4(anim,"multidim3.mp4")
```
