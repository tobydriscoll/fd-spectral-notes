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

# Fourth-order PDEs

```{code-cell}
include("smij-functions.jl");
```

Consider the **beam equation**

$$
\partial_{xxxx} u = f(x), \quad u(\pm1) = \partial_x u(\pm1) = 0. 
$$

These are known as *clamped* boundary conditions. One way to approach this problem is to introduce $v=u_x$ and solve the system

$$
\partial_x u - v &= 0, \\ 
\partial_{xxx} v &= f,
$$

with both $u$ and $v$ zero at the endpoints.

### p38: solve $u_{xxxx} = e^x,\; u(-1)=u(1)=u'(-1)=u'(1)=0$

```{code-cell}

# Construct discrete biharmonic operator:
N = 25

D, x = cheb(N)
D³ = (D^3)[2:N, 2:N]
B = [ D[2:N, 2:N] -I; zeros(N-1,N-1) D³ ]

f = @. exp(x[2:N])
rhs = [zeros(N-1); f]

# Solve boundary-value problem and plot result:
w = B \ rhs
u = [0; w[1:N-1]; 0]


# clf()
# axes([0.1, 0.4, 0.8, 0.5])
# plot(x, u, ".", markersize=10)
# axis([-1, 1, -0.01, 0.06])
# grid(true)
# xx = (-1:0.01:1)
# uu = (1 .- xx .^ 2) .* fit(x, S * u).(xx)
# plot(xx, uu)

# # Determine exact solution and print maximum error:
# A = [1 -1 1 -1; 0 1 -2 3; 1 1 1 1; 0 1 2 3]
# V = xx .^ (0:3)'
# c = A \ exp.([-1, -1, 1, 1])
# exact = exp.(xx) - V * c
# title("max err = $(round(norm(uu-exact,Inf),sigdigits=5))", fontsize=12)
```

```{code-cell}
using CairoMakie
lines(x,u)
```

```{code-cell}

```
