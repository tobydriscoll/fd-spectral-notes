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

# Absolute stability


## Step size selection

```{code-cell}
using OrdinaryDiffEq, Plots
function predprey(u,params,t)
    rabbits,foxes = u 
    ⍺,β,γ,δ = params
    ∂ₜrabbits = ⍺*rabbits - β*rabbits*foxes
    ∂ₜfoxes = γ*rabbits*foxes - δ*foxes
    return [∂ₜrabbits,∂ₜfoxes]
end

u₀ = [40,2]
tspan = (0.0,100.0)
ivp = ODEProblem(predprey,u₀,tspan,(0.2,0.1,0.05,0.3))
sol = solve(ivp,BS5())

t = sol.t
Δt = [t[k+1]-t[k] for k in 1:length(t)-1]
plot(layout=(2,1),link=:x)
plot!(sol,subplot=1)
plot!(t[1:end-1],Δt,subplot=2)
```
