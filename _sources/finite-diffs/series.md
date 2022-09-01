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
  display_name: SageMath 9.6
  language: sage
  name: sagemath-9.6
---

# Power series and finite differences

A standard way to uncover the accuracy of a finite-difference formula is by expansion in power series. 

:::{prf:definition} Truncation error for finite differences
If $d_h(u)$ is a finite-difference approximation to $u'(0)$, the **truncation error** of the approximation is 

$$
T(h) = d_h(u) - u'(0). 
$$

The power of $h$ in the leading term of $T(h)$ is the **order of accuracy** of the FD formula. 
:::

::::{prf:example}
For the forward-difference formula applied to a smooth $u(x)$,

$$
T(h) &= \frac{u(h)-u(0)}{h} - u'(0) \\ 
&= \frac{hu'(0) + \tfrac{1}{2}h^2u''(0) + \cdots}{h} - u'(0) = \tfrac{1}{2}hu''(0) + O(h^2). 
$$

Thus, it is a first-order method.
::::

There is a lot of boilerplate in these calculations that we can condense using formal power series. Suppose $hD$ represents the differentiation operator times the step size $h$. Then define $Z=\exp(hD)$. In Sage, this is done via the following:

```{code-cell} ipython3
R.<hD> = QQ[[]]
ser = hD + O(hD^8)
Z = ser.exp()
Z
```

We can interpret Taylor's theorem as stating that $Z$ is an operator that *shifts* a smooth function $u(x)$ to the function $u(x+h)$. Thus, for example, the forward difference formula is represented by the series

```{code-cell} ipython3
fd1 = Z - 1
fd1
```

Hence the truncation error is 

$$
\frac{Z-1}{h} - D = \frac{1}{2}h D^2 + O(h^2),
$$

which restates the example above. Similarly, the two-term backward difference is also first-order accurate:

```{code-cell} ipython3
bd1 = 1 - Z^(-1) 
bd1
```

The centered difference, however, is second-order accurate:

```{code-cell} ipython3
cd2 = (Z - Z^(-1))/2
cd2
```

$$
u(x+h) - u(x) & = hu'(x) + \frac{1}{2}h^2u''(x) + \cdots \\ 
\frac{u(x+h)-u(x-h)}{2h} - u'(x) &= \frac{1}{6} h^2 u'''(x) + O(h^4).
$$

One interpretation is that the antisymmetry around $h=0$ buys us an extra order.

## Derivations

The power series analysis can be used to derive new FD methods on equispaced grids. You decide which nodes to include (i.e., the **stencil** of the method), give each function value an unknown coefficient, expand everything around the evaluation point, and choose the coefficients to cancel out as many terms as possible. 

There is a more direct method, though. Formally, $Z=e^{hD}$ around $h=0$, so that $hD = \log(Z)$ around $Z=1$. Hence, finite difference formulas are found by truncating expansions of $\log(z)$ around $z=1$:

```{code-cell} ipython3
var('z')
taylor(log(z),z,1,4)
```

If we didn't already know the two-point forward difference formula, we'd start with

$$
\frac{a_1 u(h) + a_0 u(0)}{h} & \approx u'(0), \\ 
a_1 Zu(0) + a_0u(0) & \approx hDu(0), \\ 
a_1 Z + a_0 & \approx \log(Z). 
$$

Hence,

```{code-cell} ipython3
expand(taylor(log(z),z,1,1))
```

If we find a formula using values at $0$, $h$, and $2h$ to get $u'(0)$, then we have $a_2 Z^2 + a_1 Z + a_0 \approx \log(Z)$:

```{code-cell} ipython3
expand(taylor(log(z),z,1,2))
```

This corresponds to a 2nd-order forward difference on three points:

$$
\frac{-3u(0) + 4u(h) - u(2h)}{2h} = u'(0) + O(h^2). 
$$

We can derive centered, backward, and other formulas by generalizing the trick just a little. In the centered three-point formula, we have 

$$
\frac{a_{-1}u(-h) + a_0 u(0) + a_1u(h)}{h} & \approx u'(0)  \\ 
a_{-1}Z^{-1} + a_0 + a_1 Z & \approx \log(Z) \\ 
a_{-1} + a_0 Z + a_1 Z^2 & \approx Z \log(Z) 
$$

```{code-cell} ipython3
expand(taylor(z*log(z),z,1,2))
```

Similarly, we can derive a five-point centered formula via

```{code-cell} ipython3
expand(taylor(z^2*log(z),z,1,4))
```

## Higher derivatives

We can play the same games for formulas for the 2nd and higher derivatives. Here is the most popular formula for a second derivative: centered, using three nodes:

```{code-cell} ipython3
expand(taylor(z*log(z)^2,z,1,3))
```

We can verify that the formula is second-order accurate:

```{code-cell} ipython3
(Z^2-2*Z+1)/Z
```
I.e.,

$$
u''(0) = \frac{u_{-1}-2u_0+u_1}{h^2} - \frac{1}{12}h^2 u^{(4)}(0) + O(h^4).
$$

This can be carried out to 4th-order using 5 points:

```{code-cell} ipython3
expand(taylor(z^2*log(z)^2,z,1,5))
```

```{code-cell} ipython3
(-1/12*Z^4 + 4/3*Z^3 - 5/2*Z^2 + 4/3*Z - 1/12)/Z^2
```

We can also do asymmetric (forward and backward) formulas:

```{code-cell} ipython3
expand(taylor(log(z)^2,z,1,3))
```

I.e.,

$$
u''(0) = \frac{2u_{0}-5u_1+4u_2 - u_3}{h^2} + O(h^2).
$$