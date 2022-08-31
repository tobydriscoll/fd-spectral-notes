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

We're moving on to time-dependent problems, which end up being reduced to systems of ODEs in time. So we need a refresher on solution methods for IVPs,

$$
\partial_t \bfu = \bff(t,\bfu), \quad a \le t \le b, \quad \bfu(a) = \bfu_0. 
$$

All the methods aim to produce an increasing sequence of times $t_0=a,t_1,t_2,\ldots,t_n=b$ and a sequence of corresponding approximate solution values $\bfu_0,\ldots,\bfu_n$. Unlike a BVP, in an IVP there is an "arrow of time" that says the future is determined by the past, and not the other way around. Because of this, IVP solvers are **marching methods** that start from the initial condition and compute the solution values sequentially.

For example, here is a solution of the scalar problem $\dot{u}=t+u$, $u(0)=1$, using a well-known RK4 method.

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
scatter(sol.t,sol.u)
```

Even though the solution is computed only at selected values of $t$, most modern solvers provide interpolation methods that allow you to evaluate the solution anywhere you'd like.

```{code-cell}
sol(0.1),sol(0.5)
```

```{code-cell}
plot(sol,label="")
```

Note that we had to write a function of not just `u` and `t` but also a third argument, `p`. Its role is to allow the specification of parameters that are constant during one IVP solution but may be varied between different versions of the problem. 

Here is the famous Lotka-Volterra system.

```{code-cell}
function predprey(u,params,t)
    rabbits,foxes = u 
    ⍺,β,γ,δ = params
    ∂ₜrabbits = ⍺*rabbits - β*rabbits*foxes
    ∂ₜfoxes = γ*rabbits*foxes - δ*foxes
    return [∂ₜrabbits,∂ₜfoxes]
end

u₀ = [40,2]
tspan = (0.0,150.0)
ivp = ODEProblem(predprey,u₀,tspan,(0.2,0.1,0.05,0.3))
sol = solve(ivp,BS5())
```

```{code-cell}
plot(sol,label=["rabbit" "fox"])
```

For larger problems, it's more efficient to write the time-derivative function in a *mutating* manner. It's not really any harder for us to do:

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

