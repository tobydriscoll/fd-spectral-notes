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

# Two-point boundary-value problem


$$
\begin{align}
u''(x) + p(x)u'(x) + q(x)u(x) &= f(x), \quad a < x < b, \\ 
u(a)  &= \alpha, \\ 
u(b)  &= \beta.
\end{align}
$$


$$
A u = f, \qquad A = D_{xx} + P D_{x} + {Q}
$$


$$
u_0 = \alpha, \, u_n = \beta. 
$$


```{code-cell}
using LinearAlgebra
function diffmats(a,b,n)
    # assumes evenly spaced nodes
    h = (b-a)/n
    x = [a + i*h for i in 0:n]
    Dx = 1/2h*diagm(-1=>[fill(-1.,n-1);-2],0=>[-2;zeros(n-1);2],1=>[2;ones(n-1)])
    Dxx = 1/h^2*diagm(-1=>[ones(n-1);-2],0=>[1;fill(-2.,n-1);1],1=>[-2;ones(n-1)])
    Dxx[n+1,n-1] = Dxx[1,3] = 1/h^2
    return x,Dx,Dxx
end

x,Dx,Dxx = diffmats(0,1,5)
Dxx
```

```{code-cell}
include("diffmats.jl")
n = 300
x,Dx,Dxx = diffmats(0,1,n)
A = Dxx + I
A[[1,n+1],:] .= 0

A[1,1] = 1; A[n+1,n+1] = 1;
println("cond = $(cond(A))")
```

```{code-cell}
A[1,1] = n^2; A[n+1,n+1] = n^2;
println("cond = $(cond(A))")
```


## Advection-diffusion


$$
\partial_t u + c \partial_x u = k \partial_{xx} u, \quad c, k \ge 0. 
$$

$$
\text{Pe} = \frac{c}{k} =:\lambda^{-1}. 
$$

$$
\partial_x u = \lambda \partial_{xx} u
$$

$$u(-1)=1,\quad u(1)=-1$$

```{code-cell}
function advdiff(a,b,λ,n)
    x,Dx,Dxx = diffmats(a,b,n)
    A = Dx - λ*Dxx
    A[[1,n+1],:] .= 0
    A[1,1] = A[n+1,n+1] = n^2
    f = [n^2; zeros(n-1); -n^2]
    return x,A\f 
end
```

```{code-cell}
using Plots
default(size=(500,240))
plt = plot(legend=:bottomleft)
for λ in [10,1,0.5,0.1]
    x,u = advdiff(-1,1,λ,400)
    plot!(x,u,label="λ=$λ")
end
plt
```

```{code-cell}
a = 0; b = 1; n = 300;
x,Dx,Dxx = diffmats(a,b,n)
A = Dxx + I
Ã = A[2:n,2:n]
cond(Ã)
```

```{code-cell}
function advdiff(a,b,λ,n)
    x,Dx,Dxx = diffmats(a,b,n)
    A = Dx - λ*Dxx
    f = zeros(n+1)
    f = f - (1)*A[:,1] - (-1)*A[:,n+1]
    Ã = A[2:n,2:n]
    f̃ = f[2:n]
    ũ = Ã\f̃
    cond(Ã)
    return x,[1;ũ;-1]
end
```

## Convergence


```{code-cell}
λ = 1
n = [10*2^m for m in 1:8]
sol = []
for n in n
    x,u = advdiff(-1,1,λ,n)
    push!(sol,u)
end
```

```{code-cell}
sol[2]
```

```{code-cell}
û = sol[end]
err = []
for (i,u) in pairs(sol)
    push!(err,norm(û[1:2^(8-i):end]-u)/norm(u))
end

using PrettyTables
pretty_table((n=n,error=err,ratio=[NaN;err[1:end-1]./err[2:end]]))
```

```{code-cell}

```
