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
\ell_i(x) = \frac{\Phi(x)}{\Phi'(x_i)(x-x_i)} = \frac{(x-x_0)\cdots(x-x_{i-1})(x-x_{i+1})\cdots(x-x_n)}{(x_i-x_0)\cdots(x_i-x_{i-1})(x_i-x_{i+1})\cdots(x_i-x_n)},
$$

where $\Phi(x)=\prod (x-x_i)$ is a polynomial we will keep in our pocket for later. This famous formula is not a good one to use directly for numerical computation, but it does lead to some recursive definitions that can be used to derive a general finite-difference formula. That is, given nodes $x_i$ for $i=0,\ldots,n$ and a derivative order $m$, we can find the **weights** $w_i$ such that

$$
\sum_{i=0}^n w_i u(x_i) \approx u^{(m)}(0)
$$

with order of accuracy at least $n-m+1$. The algorithm is due to Fornberg.

```{code-cell} julia
"""
    fdweights(t,m)

Compute weights for the `m`th derivative of a function at zero using
values at the nodes in vector `t`.
"""
function fdweights(t,m)
    # Recursion for one weight. 
    function weight(t,m,r,k)
        # Inputs
        #   t: vector of nodes 
        #   m: order of derivative sought 
        #   r: number of nodes to use from t 
        #   k: index of node whose weight is found

        if (m<0) || (m>r)        # undefined coeffs must be zero
            c = 0
        elseif (m==0) && (r==0)  # base case of one-point interpolation
            c = 1
        else                     # generic recursion
            if k<r
                c = (t[r+1]*weight(t,m,r-1,k) -
                    m*weight(t,m-1,r-1,k))/(t[r+1]-t[k+1])
            else
                numer = r > 1 ? prod(t[r]-x for x in t[1:r-1]) : 1
                denom = r > 0 ? prod(t[r+1]-x for x in t[1:r]) : 1
                β = numer/denom
                c = β*(m*weight(t,m-1,r-1,r-1) - t[r]*weight(t,m,r-1,r-1))
            end
        end
        return c
    end
    r = length(t)-1
    w = zeros(size(t))
    return [ weight(t,m,r,k) for k=0:r ]
end;
```


```{code-cell} julia
using LinearAlgebra
nodes = [-3,-1.25,0,1,1.9]
w = fdweights(nodes,2)  # 2nd derivative weights
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

## Hermite error formula

Another valuable byproduct of the connection to interpolation is the **Hermite error formula**. Consider first the cardinal function 

$$
\ell_i(x) = \frac{\Phi(x)}{\Phi'(x_i)(x-x_i)}. 
$$

We'll go into the complex plane with a closed contour $C_i$ that encloses $x_i$ but not any other node, nor the fixed (for now) point $x$. Cauchy tells us that

$$
\ell_i(x) = \frac{1}{2\pi i} \oint_{C_i} \frac{\Phi(x)}{\Phi'(z)(x-z)} \, dz,
$$

since the integrand is meromorphic with a simple pole at $z=x_i$. More generally, if $f(z)$ is analytic on and inside $C_i$, then

$$
\ell_i(x)f(x_i) = \frac{1}{2\pi i} \oint_{C_i} \frac{\Phi(x)f(z)}{\Phi'(z)(x-z)} \, dz.
$$

Now suppose $C$ is a closed contour enclosing all of the nodes, but not the point $x$. Then we extend the logic above to get 

$$
P(x) = \sum_i \ell_i(x)f(x_i) = \frac{1}{2\pi i} \oint_{C} \frac{\Phi(x)f(z)}{\Phi'(z)(x-z)} \, dz,
$$

which is a lovely expression for the interpolating polynomial $P$. Finally, if we extend the contour to include $x$ as well, we get one more pole at $x$ and the following result.

```{prf:theorem} Hermite error formula
If $\Gamma$ is a simple closed contour enclosing all the nodes $x_0,\ldots,x_n$ and the real point $x$, $f$ is analytic on and inside $\Gamma$, and $P$ is the Lagrange interpolating polynomial for $f$ at the nodes, then 

$$
P(x) - f(x) = \frac{1}{2\pi i} \oint_{\Gamma} \frac{\Phi(x)f(z)}{\Phi'(z)(x-z)} \, dz. 
$$
```

For the special case of equally spaced nodes, this formula can be used to produce order bounds on derivative errors $P'(x)-f'(x)$ at the nodes, but we won't pursue these.