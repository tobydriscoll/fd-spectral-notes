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

# Nonlinear TPBVP


$$
\lambda u'' + u(u'-1) = -x, \quad u(0)=-1, \quad u(1)=1. 
$$

```{code-cell}
include("diffmats.jl")
using NLsolve
function bvp(λ,n)
  a,b = 0,1
  x,Dx,Dxx = diffmats(a,b,n)
  ode = u -> λ*Dxx*u + u.*(Dx*u .- 1) + x
  residual = u -> [ode(u)[2:n];u[1]+1;u[n+1]-1]
  u = nlsolve(residual,zeros(n+1))
  return x,u.zero
end;
```

```{code-cell}
using Plots
plt = plot(legend=:topleft)
for λ in [0.2,0.05,0.01]
    x,u = bvp(λ,300);
    plot!(x,u,label="λ=$λ")
end
plt
```

### Exact Jacobian


$$
u'' + u u' - 1 = 0,
$$

subject to Dirichlet boundary conditions at $x=0$ and $x=1$.

```{code-cell}
include("diffmats.jl")
n = 200
h = 1/n
x,Dx,Dxx = diffmats(0,1,n)

function residual(u) 
    r = Dxx*u + u.*(Dx*u) .- 1
    r[1] = (-1.5u[1]+2u[2]-0.5u[3])/h
    r[n+1] = u[n+1]-2
    return r
end;

function jac(u)
    ux,uxx = Dx*u,Dxx*u
    J = Dxx + diagm(ux) + diagm(u)*Dx
    # Neumann on left, Dirichlet on right 
    J[1,:] .= [-1.5/h;2/h;-0.5/h;zeros(n-2)]
    J[n+1,:] .= [zeros(n);1]
    return J
end;
```

```{code-cell}
u = x.^2 .+ 1
v = cos.(π*x/2)
ϵ = 1e-5
du = (residual(u+ϵ*v) - residual(u))/ϵ
Jv = jac(u)*v
norm(du-Jv)
```

```{code-cell}
u = 0*x
for k in 1:6
    du = jac(u)\residual(u)
    println("norm du = $(norm(du))")
    u -= du 
end

using Plots
default(size=(500,240))
plot(x,u,title="residual norm = $(norm(residual(u)))",label="")
```

Most practical nonlinear solvers will happily accept a function to compute the exact Jacobian:

```{code-cell}
sol = nlsolve(residual,jac,0*x)
plot(x,sol.zero,title="residual norm = $(sol.residual_norm)")
```

### Finite differences

![dawg](../../../../tpbvp/yo-dawg-fd.jpg)

+++

## Continuation

**Allen-Cahn equation**:

$$
\lambda u'' + u - u^3 = -\sin(5x), \quad u(0)=-1, \quad u(1)=1. 
$$

```{code-cell}
n = 200
a,b = 0,1
λ = 0.08
x,Dx,Dxx = diffmats(a,b,n)
ode = u -> λ*Dxx*u + u - u.^3 + sin.(5x)
residual(u) =[ode(u)[2:n];u[1]+1;u[n+1]-1]
@elapsed sol1 = nlsolve(residual,zeros(n+1))
```

```{code-cell}
plot(x,sol1.zero,label="λ=0.08",leg=:bottomright)
```

```{code-cell}
λ = 0.07
@elapsed sol2 = nlsolve(residual,zeros(n+1))
```

```{code-cell}
plot!(x,sol2.zero,label="λ=0.07?!")
```

```{code-cell}
(sol1.residual_norm,sol2.residual_norm)
```

```{code-cell}
sol2
```

```{code-cell}
@elapsed sol2 = nlsolve(residual,sol1.zero)
```

```{code-cell}
plot!(x,sol2.zero,label="λ=0.07")
```

```{code-cell}

```
