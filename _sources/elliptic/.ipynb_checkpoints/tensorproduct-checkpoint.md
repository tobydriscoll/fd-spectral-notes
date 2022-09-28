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

# Tensor-product grids

```{code-cell}
m = 4;   x = range(0,2,m+1);
n = 2;   y = range(1,3,n+1);
```

```{code-cell}
f = (x,y) -> cos(π*x*y-y)
F = [ f(x,y) for x in x, y in y ]
```

<!-- The plots of this section look better using a different graphics engine on the back end: -->

```{code-cell}
using Plots
```

```{code-cell}
m = 60;   x = range(0,2,m+1);
n = 48;   y = range(1,3,n+1);
F = [ f(x,y) for x in x, y in y ];

plot(x,y,F',levels=10,fill=true,aspect_ratio=1,
    color=:bluesreds,clims=(-1,1),
    xlabel="x",ylabel="y")
```

```{code-cell}
surface(x,y,F',l=0,leg=:none,
    color=:bluesreds,clims=(-1,1),
    xlabel="x",ylabel="y",zlabel="f(x,y)")
```

## Parameterized surfaces


```{code-cell}
r = range(0,1,length=41)
θ = range(0,2π,length=81)
F = [ 1-r^4 for r in r, θ in θ ]

surface(r,θ,F',legend=:none,l=0,color=:viridis,
    xlabel="r",ylabel="θ",title="A polar function")
```

```{code-cell}
X = [ r*cos(θ) for r in r, θ in θ ]
Y = [ r*sin(θ) for r in r, θ in θ ]

surface(X',Y',F',legend=:none,l=0,color=:viridis,
    xlabel="x",ylabel="y",title="Function on the unit disk")
```

```{code-cell}
θ = range(0,2π,length=61)
ϕ = range(0,π,length=51)

X = [ cos(θ)*sin(ϕ) for θ in θ, ϕ in ϕ ]
Y = [ sin(θ)*sin(ϕ) for θ in θ, ϕ in ϕ ]
Z = [ cos(ϕ) for θ in θ, ϕ in ϕ ]

F =  @. X*Y*Z^3
surface(X',Y',Z',fill_z=F',l=0,leg=:none,color=:viridis,
    xlims=(-1.1,1.1),ylims=(-1.1,1.1),zlims=(-1.1,1.1),
    xlabel="x",ylabel="y",zlabel="z",
    title="Function on the unit sphere")
```

```{code-cell}
u = (x,y) -> sin(π*x*y-y);
∂u_∂x = (x,y) -> π*y*cos(πx*y-y);
∂u_∂y = (x,y) -> (π*x-1)*cos(π*x*y-y);
```

```{code-cell}
foreach(println,readlines("diffmats.jl"))
```

```{code-cell}
include("diffmats.jl")
x,Dx,_ = diffmats(m,0,2)
y,Dy,_ = diffmats(n,1,3)
mtx = (f,x,y) -> [ f(x,y) for x in x, y in y ]
U = mtx(u,x,y)
∂xU = Dx*U;
∂yU = U*Dy';
```

```{code-cell}
M = maximum(abs,∂yU)    # find the range of the result
plot(layout=(1,2),aspect_ratio=1,clims=(-M,M),xlabel="x",ylabel="y")
contour!(x,y,mtx(∂u_∂y,x,y)',layout=(1,2),levels=12,
    fill=true,color=:bluesreds,title="∂u/∂y")      
contour!(x,y,∂yU',subplot=2,levels=12,
    fill=true,color=:bluesreds,title="approximation")
```

```{code-cell}
exact = mtx(∂u_∂y,x,y)
# Relative difference in Frobenius norm:
norm(exact-∂yU) / norm(exact)
```
