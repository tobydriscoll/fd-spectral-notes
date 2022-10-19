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
  display_name: Julia 1.8.2
  language: julia
  name: julia-1.8
---

# von Neumann analysis

$$
\partial_t u + c \partial_x u = 0
$$


## Centered in space

### Euler

```{code-cell}
include("/Users/driscoll/817/notes/advection/diffmats.jl")
using OrdinaryDiffEq

n = 100
c = -1
u₀ = x -> exp(2*sin(2π*x))

x,Dx,_ = diffmats(n,0,1,periodic=true)
advect(u,c,t) = -c*(Dx*u)

ivp = ODEProblem(advect,u₀.(x),(0.,10.),c)
sol = solve(ivp,Euler(),dt=1/500,adaptive=false);
```

```{code-cell}
using Plots,PyFormattedStrings
anim = @animate for t in range(0,8,201)
    plot(x,sol(t),label=f"t={t:.1f}",m=2,
        xaxis=("x"),yaxis=("u(x,t)",[-3,9]),dpi=100)
end
mp4(anim,"advect1.mp4")
```

```{code-cell}
plot(x,sol(7),m=2)
```

```{code-cell}
scatter(eigvals(-c*Matrix(Dx)),m=3,label="",aspect_ratio=1)
```

### Trapezoid

```{code-cell}
sol = solve(ivp,Trapezoid(),dt=1/20,adaptive=false);
anim = @animate for t in range(0,8,201)
    plot(x,sol(t),label=f"t={t:.1f}",m=2,
        xaxis=("x"),yaxis=("u(x,t)",[-3,9]),dpi=100)
end
mp4(anim,"advect2.mp4")
```

## Upwind in space

```{code-cell}
h = 1/n
Dx = (1/h)*spdiagm(0=>-ones(n),1=>ones(n-1),1-n=>[1.])
sol1 = solve(ivp,Euler(),dt=h/abs(c)*0.95,adaptive=false)
sol2 = solve(ivp,Euler(),dt=h/abs(c)*1.05,adaptive=false)

anim = @animate for t in range(0,5,201)
    plot(x,sol1(t),label="stable",
        xaxis=("x"),yaxis=("u(x,t)",[-3,9]),dpi=100)
    plot!(x,sol2(t),label="unstable",m=2)
end
mp4(anim,"advect3.mp4")
```

```{code-cell}
scatter(eigvals(-c*1.05h*Matrix(Dx)),m=3,label="",aspect_ratio=1)
```
