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

# Semidiscretization


$$
\partial_t u = D_{xx} u, \quad u(0) = [ u_0(x_i) ]_i, 
$$

## Stiffness


```{code-cell}
using OrdinaryDiffEq, SparseArrays, BenchmarkTools, LinearAlgebra

n = 40
h = 1/n
x = [ i*h for i in 0:n-1 ]
Dxx = 1/h^2*spdiagm(-1=>ones(n-1),
    0=>fill(-2.,n),
    1=>ones(n-1),
    1-n=>[1.],
    n-1=>[1.])

u₀(x) = exp(sin(3π*x))
ivp = ODEProblem((u,p,t)->Dxx*u,u₀.(x),(0,0.1));
sol = solve(ivp,Euler(),dt=1/1000)
norm(sol.u[end])
```

```{code-cell}
n = 300
h = 1/n
x = [ i*h for i in 0:n-1 ]
Dxx = 1/h^2*spdiagm(-1=>ones(n-1),
    0=>fill(-2.,n),
    1=>ones(n-1),
    1-n=>[1.],
    n-1=>[1.])

u₀(x) = exp(sin(3π*x))
ivp = ODEProblem((u,p,t)->Dxx*u,u₀.(x),(0,0.1));
```

```{code-cell}
@time sol = solve(ivp,RK4(),abstol=1e-7,reltol=1e-7)
println("$(length(sol.t)) time steps taken")
```

```{code-cell}
using Plots,PyFormattedStrings
anim = @animate for t in range(0,0.1,40)
    plot(x,sol(t),ylims=(0,2.8),label="",
        title=f"t = {t:.3f}",size=(500,260),dpi=150)
end
mp4(anim,"diffusion.mp4")
```

```{code-cell}
τ = maximum(diff(sol.t))

τ,1/n^2
```

```{code-cell}
gr()
λ = eigvals(Matrix(Dxx))

scatter(real(τ*λ),imag(τ*λ),aspect_ratio=1,label="",
    xlabel="Re ζ",ylabel="Im ζ")
```

```{code-cell}
@time sol = solve(ivp,Rodas4P(),abstol=1e-7,reltol=1e-7);
println("$(length(sol.t)) time steps taken")
```

```{code-cell}

```
