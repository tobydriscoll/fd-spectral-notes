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

# Chebyshev differentiation matrix

Say we are given a grid function $v$ on the Chebyshev points

$$
x_j = \cos\left( \frac{j\pi}{N} \right), \qquad j=0,\ldots,N. 
$$

```{caution}
The definition above orders the Chebyshev points from right to left, which can cause confusion. Note also that there are $N+1$ of them when we refer to degree $N$ interpolants.
```

The Chebyshev spectral collocation scheme is:

1. Let $p$ be the unique polynomial of degree no more than $N$ interpolating $v$ at the $x_j$. 
2. Set $w_j=p'(x_j)$ for all $j$.

The process is linear, so there is a matrix $\bfD_N$ such that

$$
\bfw = \bfD_N \bfv. 
$$

```{note}
The restriction to even $N$ for our Fourier formulas does not apply to the Chebyshev formulas we present.
```

Unlike the Fourier case, we do not have translation invariance along the Chebyshev grid, so more than one column of $\bfD_N$ has to be worked out.

::::{prf:example}
For $N=2$, we have $x_0=1$, $x_1=0$, and $x_2=-1$, and we can write 

$$
p(x) &= \frac{(x)(x+1)}{(1)(1+1)}v_0 + \frac{(x-1)(x+1)}{(-1)(+1)}v_1 + \frac{(x-1)(x)}{(-1-1)(-1)}v_2 \\ 
&= \frac{1}{2}(x^2+x)v_0 + (1-x^2)v_1 + \frac{1}{2}(x^2-x)v_2. 
$$

Thus,

$$
p'(x) = \frac{1}{2}(2x+1)v_0 -2x v_1 + \frac{1}{2}(2x-1)v_2, 
$$

and we get

$$
\bfD_2 = \begin{bmatrix}
  \tfrac{3}{2} & -2 & \tfrac{1}{2} \\ 
  \tfrac{1}{2} & 0 & -\tfrac{1}{2} \\ 
  -\tfrac{1}{2} & 2 & -\tfrac{3}{2}
\end{bmatrix}. 
$$

We have seen these rows of numbers occur before, since they arise from finite differences on 3 equally spaced points.
::::

Formulas for the entries of $\bfD_N$ in the general case are given in the textbook on p. 53. Here is a code that implements them:

```{code-cell} julia
using Sugar, SpectralMethodsTrefethen
Sugar.get_source(first(methods(cheb))) |> last |> print
```

The code above does not use formulas for the diagonal entries. Instead, it uses the *negative sum trick*, which arises from the fact that the derivative of a constant function is exactly zero in a spectral method, and rewrites the condition $\sum_j (D_N){ij} = 0$ as an equation for the diagonal term.

Note that the differentiation matrix does have the antisymmetry property

$$
(D_N){N-i,N-j} = -(D_N)_{i,j}. 
$$

Otherwise, the matrices do not have much obvious structure:

```{code-cell} julia
D, _ = cheb(4)
D
```

```{code-cell} julia
D, _ = cheb(5)
D
```

For the analytic, nonperiodic function $e^x\sin(5x)$, the spectral derivative has about 2 accurate digits at $N=10$ and 9 at $N=20$:

```{code-cell} julia
Sugar.get_source(first(methods(p11))) |> last |> print
p11()
```

The effects of smoothness are illustrated more clearly here:

```{code-cell} julia
Sugar.get_source(first(methods(p12))) |> last |> print
p12()
```

The function $|x|^3$ has only 2 continuous derivatives, so the convergence is algebraic. Next, $e^{-x^2}$ is not analytic, and its convergence is between algebraic and exponential. The case $1/(1+x^2)$ is analytic around the interval, showing spectral convergence, and the function $x^{10}$ is the polynomial analog of "band-limited."