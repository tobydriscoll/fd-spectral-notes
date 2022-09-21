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

# Resolving layers

$$
\partial_x u - \lambda \partial_{xx} u = 0
$$

subject to $u(-1)=1$, $u(1)=-1$.

```{code-cell}
include("diffmats.jl")
function advdiff(λ,n)
    a,b = -1,1
    x,Dx,Dxx = diffmats(a,b,n)
    Ã = Dx - λ*Dxx
    A = diagm(ones(n+1))
    A[2:n,:] .= Ã[2:n,:]
    f = [1; zeros(n-1); -1]
    return x,A\f 
end;
```

```{code-cell}
using Plots
default(size=(500,240))
plt = plot(legend=:bottomleft)
for λ in [1,0.1,0.05,0.01,0.005]
    x,u = advdiff(λ,100)
    plot!(x,u,label="λ=$λ")
end
plt
```

## Change of coordinate


```{code-cell}
plot(tan,-1.55,0)
```

```{code-cell}
Plots.default(label="")
M = 30
γ = atan(M)
ξ(s) = 1 + 2/M*tan(γ*(s-1)/2)
plot(ξ,-1,1,aspect_ratio=1,xlabel="s",ylabel="x")

s = range(-1,1,41)
x = ξ.(s)
plot!([-1.5ones(41) s]',[x x]',color=Gray(0.7),m=2)
```

```{code-cell}
function advdiff(λ,M,n)
    a,b = -1,1
    s,Ds,Dss = diffmats(a,b,n)
    γ = atan(M)
    s₁ = @. γ*(s-1)/2
    x = @. 1 + 2/M*tan(s₁)
    ξʹ = @. (γ/M)*sec(s₁)^2
    ξʹʹ = @. (γ^2/M)*sec(s₁)^2*tan(s₁)
    Dx = diagm(@. 1/ξʹ)*Ds 
    Dxx = diagm(@. 1/(ξʹ)^2)*Dss - diagm(@. ξʹʹ/(ξʹ)^3)*Ds

    A = Dx - λ*Dxx
    A[[1,n+1],:] .= I(n+1)[[1,n+1],:]
    f = [1; zeros(n-1); -1]
    return s,x,A\f 
end
```

```{code-cell}
plt = plot(legend=:bottomleft)
for λ in [1,0.1,0.05,0.01,0.001]
    s,x,u = advdiff(λ,50,100)
    plot!(x,u,label="λ=$λ",xlabel="x")
end
plt
```

```{code-cell}
plt = plot(legend=:bottomleft)
for λ in [1,0.1,0.05,0.01,0.005]
    s,x,u = advdiff(λ,50,100)
    plot!(s,u,label="λ=$λ",xlabel="s")
end
plt
```

```{code-cell}

```
