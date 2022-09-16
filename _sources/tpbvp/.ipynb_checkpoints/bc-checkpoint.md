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

# Boundary conditions for the TPBVP

We return to the linear TPBVP,

$$
\begin{align}
u''(x) + p(x)u'(x) + q(x)u(x) &= f(x), \quad a < x < b, \\ 
G_{11}u(a) + G_{12}u'(a)  &= \alpha, \\ 
G_{21}u(b) + G_{22}u'(b)  &= \beta,
\end{align}
$$

## Fictitious points


$$
u''(x_0) \approx \frac{u_{-1}-2u_0+u_1}{h^2}
$$


$$
u'(x_0) \approx \frac{u_{1}-u_{-1}}{2h} 
$$


$$
G_{11}u_0 + G_{12}\frac{u_{1}-u_{-1}}{2h}  = \alpha
$$


```{code-cell}
using LinearAlgebra
p = x -> x+2
q = x -> x^2+1
f = x -> 1
a,b = -1,1
n = 6
h = (b-a)/n
x = [a + i*h for i in 0:n];
```

```{code-cell}
Dxx = 1/h^2 * diagm(n+1,n+3,0=>ones(n+1),1=>-2*ones(n+1),2=>ones(n+1))
```

```{code-cell}
Dx = 1/2h * diagm(n+1,n+3,0=>-ones(n+1),2=>ones(n+1))
```

```{code-cell}
A = Dxx + diagm(p.(x))*Dx + diagm(n+1,n+3,2=>q.(x))
v = f.(x)
A
```

```{code-cell}
Ga,Gb = [0,1],[1,-1]
⍺,β = 0,0
r₋ = [-Ga[2]/2h Ga[1] Ga[2]/2h zeros(1,n)]
r₊ = [zeros(1,n) -Gb[2]/2h Gb[1] Gb[2]/2h]
A = [r₋;A;r₊]
v = [⍺;v;β]
A
```

```{code-cell}
A\v
```


## Schur complement


```{code-cell}
i1,i2 = 2:n+2,[1,n+3]
S = A[i1,i2]/A[i2,i2]
ṽ = v[i1] - S*v[i2]
Ã = A[i1,i1] - S*A[i2,i1]
Ã
```

```{code-cell}
u = Ã\ṽ
```

```{code-cell}
n = 200
h = (b-a)/n
x = [a + i*h for i in 0:n]
Dxx = 1/h^2 * diagm(n+1,n+3,0=>ones(n+1),1=>-2*ones(n+1),2=>ones(n+1))
Dx = 1/2h * diagm(n+1,n+3,0=>-ones(n+1),2=>ones(n+1))
A = Dxx + diagm(p.(x))*Dx + diagm(n+1,n+3,2=>q.(x))
v = f.(x)
ra = [-Ga[2]/2h Ga[1] Ga[2]/2h zeros(1,n)]
rb = [zeros(1,n) -Gb[2]/2h Gb[1] Gb[2]/2h]
A = [ra; A; rb]
v = [⍺; v; β]
i1,i2 = 2:n+2,[1,n+3]
S = A[i1,i2]/A[i2,i2]
ṽ = v[i1] - S*v[i2]
Ã = A[i1,i1] - S*A[i2,i1]
u = Ã\ṽ;

using Plots
plot(x,u,aspect_ratio=1)
```

```{code-cell}
Gb[1]*u[n+1] + Gb[2]*(u[n+1]-u[n])/h - β
```

```{code-cell}

```
