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

# Interpolation and finite differences

## Fornberg's algorithm


$$
P(x) = \sum_{i=0}^n y_i \ell_i(x), 
$$

where each $\ell_i$ is the **cardinal polynomial** 

$$
\ell_i(x) = \frac{\Phi(x)}{\Phi'(x_i)(x-x_i)} = \frac{(x-x_0)\cdots(x-x_{i-1})(x-x_{i+1})\cdots(x-x_n)}{(x_i-x_0)\cdots(x_i-x_{i-1})(x_i-x_{i+1})\cdots(x_i-x_n)},
$$

where $\Phi(x)=\prod (x-x_i)$ .

Given nodes $x_i$ for $i=0,\ldots,n$ and a derivative order $m$, we can find the **weights** $w_i$ such that

$$
\sum_{i=0}^n w_i u(x_i) \approx u^{(m)}(0)
$$


```{code-cell}
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

```{code-cell}
using LinearAlgebra
```

```{code-cell}
nodes = [0,1//1,2]
fdweights(nodes,1)
```

```{code-cell}
nodes = [-3,-1.25,0,1,1.9]
w = fdweights(nodes,2)  # 2nd derivative weights
f = x->cos(2x)
h = 0.05; 
y = f.(h*nodes)
val = dot(w,y)/h^2
err1 = val - (-4)
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
