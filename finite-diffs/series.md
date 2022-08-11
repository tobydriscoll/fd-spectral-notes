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

A standard way to uncover the accuracy of a finite-difference formula is by expansion in power series. We'll use formal power series here, which is a bit tricky. Suppose $hD$ represents the differentiation operator times a number $h$. Then define

```{code-cell} ipython3
R.<hD> = QQ[[]]
ser = hD + O(hD^6)
Z = ser.exp()
Z
```

We can interpret Taylor's theorem as stating that $Z$ is an operator that "shifts" a smooth function $u(x)$ to the function $u(x+h)$. Thus, for example, a forward difference operator has the series

```{code-cell} ipython3
fd1 = Z - 1
fd1
```

That is,

$$
u(x+h) - u(x) & = hu'(x) + \frac{1}{2}h^2u''(x) + \cdots \\ 
\frac{u(x+h)-u(x)}{h} = u'(x) + \frac{1}{2}h u''(x) + O(h^2).
$$

Hence we say this formula is **first-order accurate**, since its error is led by the $O(h^1)$ term. Similarly, the two-term backward difference is also first-order:

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
\frac{u(x+h)-u(x-h)}{2h} &= u'(x) + \frac{1}{6} h^2 u'''(x) + O(h^4).
$$

One interpretation is that the antisymmetry around $h=0$ buys us an extra order.

We can also use the connection between differentiation and shifting to derive formulas. Note that formally, $Z=e^{hD}$ around $h=0$, so that $hD = \log(Z)$ around $Z=1$. Hence finite difference formulas are always found as truncations of the series:

```{code-cell} ipython3
var('z')
taylor(log(z),z,1,4)
```

So if we were setting out to find the two-point forward difference, for example, we'd start with

$$
\frac{a_1 u(h) + a_0 u(0)}{h} & \approx u'(0) \\ 
a_1 Zu(0) + a_0u(0) & \approx hDu(0) \\ 
a_1 Z + a_0 & \approx \log(Z). 
$$

Hence,

```{code-cell} ipython3
expand(taylor(log(z),z,1,1))
```

If we add a node at $2h$, then we have $a_2 Z^2 + a_1 Z + a_0 \approx \log(Z)$:

```{code-cell} ipython3
expand(taylor(log(z),z,1,2))
```

This corresponds to a 2nd-order forward difference on three points:

$$
\frac{-3u(0) + 4u(h) - u(2h)}{2h} = u'(0) + O(h^2). 
$$

We can handle centered (and any generally-offset) formula on evenly spaced nodes. In the centered three-point formula, for instance, we have 

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

