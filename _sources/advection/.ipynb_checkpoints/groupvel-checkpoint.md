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

# Group velocity and dispersion



## Semidiscrete

### Centered 



$$
\omega(\xi) = \frac{c}{h} \sin(\xi h).
$$


$$
\omega'(\xi) = c \cos(\xi h),
$$

```{code-cell}
include("/Users/driscoll/817/notes/advection/diffmats.jl")
using OrdinaryDiffEq, Plots
```

```{code-cell}
n = 120;
h = 2π/n
println("sawtooth wavenumber is $(π/h)")
```

Here is a combination of a low-wavenumber and a high-wavenumber packet.

```{code-cell}
x,Dx,_ = diffmats(n,-π,π,periodic=true);
gauss = x -> exp(-18x^2)
u₀ = x -> gauss(x+2) + gauss(x-2).*sin(52x);
plot(x,u₀.(x))
```

When we solve the semidiscretization accurately for $c=1$, we can see how the high-wavenumber packet travels in the wrong direction:

```{code-cell}
c = 1
adv = (u,c,t) -> -c*Dx*u
ivp = ODEProblem(adv,u₀.(x),(0.,3.),c)
sol = solve(ivp,RK4(),abstol=1e-8,reltol=1e-8);

anim = @animate for t in range(0,3,121)
    plot(x,sol(t),ylims=(-1.2,1.2),label="",dpi=128)
end
mp4(anim,"groupvel1.mp4")
```

```{code-cell}
tent = @. max(0,1-abs(x))
ivp = ODEProblem(adv,tent,(0.,6.),c)
sol = solve(ivp,RK4(),abstol=1e-8,reltol=1e-8);

anim = @animate for t in range(0,6,121)
    plot(x,sol(t),ylims=(-1.2,1.2),label="",dpi=128)
end
mp4(anim,"groupvel2.mp4")
```

### Downwind 


## Fully discrete

### Backward Euler


```{code-cell}
ξh = range(-π,π,200)
ωτ = @. -1im*log(1 + 0.9im*sin(ξh))
plot(ξh,real(ωτ),label="Re",aspect_ratio=1)
plot!(ξh,imag(ωτ),label="Im")
plot!(ξh,ξh,l=(Gray(0.4),:dash),
    xaxis=("ξh",[-π,π]),yaxis=("ωτ",[-π,π]))
```


### Midpoint

```{code-cell}
ωτ = @. asin(0.9*sin(ξh))
ωτ2 = @. mod(2π-ωτ,2π) - π
scatter(ξh,[ωτ ωτ2],m=2,msw=0,aspect_ratio=1,legend=false)
plot!(ξh,ξh,l=(Gray(0.4),:dash),
    xaxis=("ξh",[-π,π]),yaxis=("ωτ",[-π,π]))
```

```{code-cell}
τ = 0.9h
init = x -> gauss(x)*sin(45x)
u₋ = @. init(x + c*τ)
u = init.(x)
anim = @animate for j in 1:100
    global u,u₋
    plot(x,u,ylims=(-1.2,1.2),label="",dpi=128)
    u₊ = u₋ + 2τ*(-c*Dx*u)
    u₋,u = u,u₊
end
mp4(anim,"groupvel3.mp4")
```

But if $|c|\tau / h > 1$, there will be nonreal solutions that mean exponential instability in time.

```{code-cell}
asin(complex(1.1))
```
