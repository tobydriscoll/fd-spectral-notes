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

# Multiple space dimensions

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
include("/Users/driscoll/817/notes/mol/diffmats.jl")
using OrdinaryDiffEq

m,n = 36,42
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
    contour(x,y,unpack(sol(t))',levels=range(0,3,31),aspect_ratio=1,dpi=100,
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
    contour(x,y,reshape(sol(t),m+1,n+1)',levels=range(0,3,31),aspect_ratio=1,dpi=100,
        clims=(0,3),fill=true,title=f"t={t:.2f}",color=:viridis)
end
mp4(anim,"multidim2.mp4")
```

## Explicit Jacobians

```{code-cell}
m,n = 60,80
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
    contour(x,y,reshape(sol(t),m+1,n+1)',levels=range(0,3,31),aspect_ratio=1,dpi=100,
        clims=(0,3),fill=true,title=f"t={t:.2f}",color=:viridis)
end
mp4(anim,"multidim3.mp4")
```

```{code-cell}

```
