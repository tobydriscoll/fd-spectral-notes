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
  display_name: Julia 1.8.0-rc3
  language: julia
  name: julia-1.8
---

# Interpolation and finite differences

An equivalent but different way to look at finite differences is *interpolate-differentiate-evaluate*. For example, consider again the humble two-point forward difference

$$
\frac{a_1 u(h) + a_0 u(0)}{h}  \approx u'(0). 
$$

If we draw the secant line through the points $(0,u(0))$ and $(h,u(h))$, then its slope is $(u(h)-u(0))/h$, and we recover the first-order formula. The three-point centered difference 

$$
\frac{a_{-1}u(-h) + a_0 u(0) + a_1u(h)}{h} \approx u'(0) 
$$

suggests interpolation by a parabola through three points:

$$
P(x) = \frac{u(-h) \cdot x(x-h) - 2 u(0)\cdot (x^2-h^2) + u(h)\cdot x(x+h)}{2h^2}. 
$$

From this, we can derive

$$
u'(0) \approx P'(0) = -\frac{1}{2h} u(-h) + 0 u(0) + \frac{1}{2h} u(h), 
$$

as well as

$$
u'(-h) \approx P'(-h) = -\frac{3}{2h} u(-h) + \frac{2}{h} u(0) - \frac{1}{2h} u(h), 
$$

which is the three-point forward formula we derived by power series. For that matter, we can also derive a centered formula for the second derivative,

$$
u''(0) \approx P''(0) = \frac{1}{h^2} u(-h) - \frac{2}{h^2} u(0) + \frac{1}{h^2} u(h),
$$

which happens to be second-order accurate.

## Fornberg's algorithm

A ton is known about interpolating polynomials. One of the landmark results is the **Lagrange interpolating form**. The polynomial of degree no more than $n$ passing through $n+1$ points $(x_i,y_i)$ for $i=0,\ldots,n$ is 

$$
P(x) = \sum_{i=0}^n y_i \ell_i(x), 
$$

where each $\ell_i$ is the **cardinal polynomial** 

$$
\ell_i(x) = \frac{\Phi(x)}{\Phi'(x_i)} = \frac{(x-x_0)\cdots(x-x_{i-1})(x-x_{i+1})\cdots(x-x_n)}{(x_i-x_0)\cdots(x_i-x_{i-1})(x_i-x_{i+1})\cdots(x_i-x_n)},
$$

where $\Phi(x)=\prod (x-x_i)$ is a polynomial we will keep in our pocket for later. This famous formula is not a good one to use directly for numerical computation, but it does lead to some recursive definitions that can be used to derive a general finite-difference formula. That is, given nodes $x_i$ for $i=0,\ldots,n$ and a derivative order $m$, we can find the **weights** $w_i$ such that

$$
\sum_{i=0}^n w_i u(x_i) \approx u^{(m)}(0)
$$

with order of accuracy at least $n-m+1$. The algorithm is due to Fornberg. We don't present it here, but we note that it is available in Julia's `FiniteDifferences` package:

```{code-cell}
using FiniteDifferences
nodes = [-3,-1.5,0,1,2]
fd = FiniteDifferenceMethod(nodes,2)  # 2nd derivative
```

```{code-cell}
using LinearAlgebra
w = fd.coefs
f = x->cos(2x)
h = 0.05; 
err1 = dot(w/h^2,f.(h*nodes)) + 4
```

```{code-cell}
h = 0.05/2; 
err2 = dot(w/h^2,f.(h*nodes)) + 4
```

```{code-cell}
log2(err1/err2)
```

```{code-cell}

```
