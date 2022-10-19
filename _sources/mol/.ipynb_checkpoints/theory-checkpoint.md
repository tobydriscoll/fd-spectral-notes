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

```{code-cell}
include("diffmats.jl")
n = 100
h = 2π/n

x = [i*h-π for i in 0:n-1]
u = @. exp(-5x^2)

using Plots,PyFormattedStrings
default(label="",size=(500,240))
plot(x,u)
```

```{code-cell}
Dxx = 1/h^2*diagm(1-n=>[1.0],-1=>ones(n-1),0=>fill(-2.0,n),1=>ones(n-1),n-1=>[1.0])
τ = 0.6h^2 

anim = @animate for i in 1:0.3/τ
    global u += τ*(Dxx*u) 
    plot(x,u,ylims=[-0.5,1.5],label=f"t={i*τ:.3f}")
end

mp4(anim,"eulerheat.mp4")
```
