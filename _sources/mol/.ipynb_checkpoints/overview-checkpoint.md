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

# Method of lines

$\dot{u}=t+u$, $u(0)=1$

```{code-cell}
using OrdinaryDiffEq

f(u,p,t) = t + u
tspan = (0.0,3.0)
ivp = ODEProblem(f,1.,tspan)
sol = solve(ivp,RK4())
[sol.t sol.u]
```

```{code-cell}
using Plots
default(size=(500,300))
scatter(sol.t,sol.u,label="")
```

```{code-cell}
sol(0.1),sol(0.5)
```

```{code-cell}
plot(sol,label="")
```

```{code-cell}
plot(sol,label=["rabbit" "fox"])
```

```{code-cell}
function predprey!(dudt,u,params,t)
    rabbits,foxes = u 
    ⍺,β,γ,δ = params
    dudt[1] = ⍺*rabbits - β*rabbits*foxes
    dudt[2] = γ*rabbits*foxes - δ*foxes
    return nothing
end

u₀ = [40,2]
tspan = (0.0,150.0)
ivp = ODEProblem(predprey!,u₀,tspan,(0.2,0.05,0.1,0.5))
sol = solve(ivp,BS5());
sol.(50:54)
```

```{code-cell}

```
