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
  display_name: Julia 1.8.2
  language: julia
  name: julia-1.8
---

# Radial basis functions

Solving on a tensor-product grid is limiting in two major ways: 

1. Geometry: not very flexible, and especially challenging in 3D
2. Dimension: number of points grows exponentially with the dimension

Methods that work without any grid are known as **meshless** or **meshfree** methods. A common category of these are **kernel methods**, where the interpolating or approximating function is in the form

$$
u(x) = \sum_{n=1}^N c_n K(x-x_n), \qquad x \in \mathbb{R}^d,
$$

where the $x_n$ are points in $\mathbb{R}^d$ called **centers**, $c_n$ are constants, and $K$ is a kernel function. A particularly interesting choice is 

$$
K(s) = \varphi\bigl(\| s \|\bigr),
$$

where $\varphi$ is known as the **basic function** and $\|\cdot\|$ is a norm, most typically the 2-norm. The resulting $K(x-x_n)$ is known as a **radial basis function**, since it is radially symmetric around its center $x_n$. 

## Basic functions

Here are some of the most common choices for the basic function. 

* **Polyharmonic spline**: $\varphi(r) = r^k$ for odd $k$, or $r^k\log(r)$ for even values of $k$. These functions have $k-1$ continuous derivatives, with higher-order discontinuities at the center points. 
* **Wendland function**: $\varphi$ is a piecewise polynomial with compact support. The number of continuous derivatives is selectable by construction.
* **Gaussian**: $\varphi(r) = \exp(-r^2)$ (smooth)
* **Multiquadric**: $\varphi(r) = \sqrt(1+r^2)$ (smooth)

Theory (again based on Fourier analysis) states that the convergence rate of the RBF interpolant depends on the smoothness of $\varphi$ in a familiar way: algebraic if the smoothness is finite, and spectral if it's infinite. However, realizing these rates in practice, particularly in the spectral case, requires severely limited circumstances.

In practice one often adds a **shape parameter** $\epsilon > 0$:

$$
u(x) = \sum_{n=1}^N c_n \varphi\bigl(\epsilon \| x-x_n \|\bigr).
$$

It's also possible, and sometimes desirable, to let $\epsilon$ vary from center to center. The shape parameter has no effect on the polyharmonic case, it controls the support radius in the Wendland case, and it acts as a local scaling in the smooth cases.



## Interpolation conditions

Given 

$$
u(x) = \sum_{n=1}^N c_n \varphi\bigl(\| x-x_n \|\bigr), 
$$

we need $N$ conditions to determine the coefficients. The standard choice is to interpolate values at the centers:

$$
f_i = u(x_i) = \sum_{n=1}^N c_n \varphi\bigl(\| x_i-x_n \|\bigr), \qquad i=1,\ldots,n,
$$

which yields the linear system $\bfA \mathbf{c} = \bff$, where 

$$
A_{ij} = \varphi\bigl(\| x_i-x_j \|\bigr).
$$

Clearly, $\bfA$ is symmetric. For some basic functions, such as the Gaussian, it's probably positive definite, as well. 

It's common to augment the RBFs with low-degree polynomial terms:

$$
u(x) = \sum_{n=1}^N c_n \varphi\bigl(\| x-x_n \|\bigr) + \sum_{m=1}^M a_m \psi_m(x), 
$$

where $\psi_1,\ldots,\psi_M$ is a basis for polynomials in $\mathbb{R}^d$ of a fixed, low degree. In order to determine the additional degrees of freedom, we add the constraints

$$
\mathbf{P}^T \mathbf{c} = \bfzero,  \qquad P_{ij} = \psi_j(x_i). 
$$

This leads to the linear system

$$
\begin{bmatrix}
  \bfA & \mathbf{P} \\ \mathbf{P}^T & \bfzero 
\end{bmatrix} 
\begin{bmatrix}
  \mathbf{c} \\ \bfa 
\end{bmatrix}
=
\begin{bmatrix}
  \bff \\ \bfzero
\end{bmatrix}, 
$$

which remains symmetric. This formulation allows exact reproduction of low-degree polynomials, which is both practically and theoretically useful. In the multiquadric case, including the constant term ensures that the system is positive definite, while the polyharmonic case usually includes at least the constant and linear polynomials. 


## Convergence

One may define the **fill distance** for domain $\Omega$ and set of centers $x_n$ as

$$
h = \sup_{x\in\Omega} \min_n \norm{x-x_n}. 
$$

It's an analog of the grid spacing or step size. At one extreme is **nonstationary** approximation, in which $\epsilon$ is fixed as $h\to 0$, or $\epsilon\to 0$ on a fixed center set. While can get spectral convergence (in an unfamiliar norm over a disappointingly small space of functions), the condition number of the linear system also grows exponentially, severely limiting the realizable accuracy.

At the other extreme, a **stationary** approximation holds $\epsilon h$ fixed as $h\to 0$. Naively, this case exhibits neither growth in the condition number nor convergence in a function space. However, it is not useless, as it allows one to tune the tradeoff between these competing effects. One can also choose to allow $\epsilon h$ to go to zero more slowly than $h$, which leads to a compromise between conditioning and approximation accuracy.

A landmark result by Platte, Trefethen, and Kuilaars is worth mentioning here.

::::{prf:theorem} Impossibility Theorem
Suppose $E$ is a compact set containing $[-1,1]$ and let $B(E)$ be the Banach space of functions that are continuous on $E$ and analytic in its interior. If $\{\phi_n\}$ is approximation process based on sampling function $f$ at $n$ equispaced points such that, for some positive $M$ and $\sigma >1$,

$$
\norm{ f - \phi_n[f]}_{[-1,1]} \le M \sigma^{-n} \norm{f}_E
$$

for all $f\in B(E)$, then the condition number of $\phi_n$ satisfies

$$
\kappa_n \ge C^n
$$

for some $C>1$ and all sufficiently large $n$.
::::

Beyond its literal statement, the theorem shows that unless one has a special set of interpolation nodes, then there is no escaping exponential ill-conditioning when there is a spectral convergence rate, even in one dimension.

## Simple cases

For the polyharmonic spline $\varphi(r)=r$ in one dimension, the approximation $u$ is simply a piecewise linear interpolant. For $\varphi(r)=r^3$ in one dimension, one gets a cubic spline interpolant, although not with the classical choices of end conditions. 

For smooth basic functions in one dimension, then as $\epsilon \to 0$ (the "flat limit") one gets the polynomial interpolant. The same is true in higher dimensions, under some minor restrictions (polynomial interpolation in more than one dimension is no joke). This result suggests that while the basic RBF method is an unstable way to get the interpolant, there is a sensible limiting value that one can try to come at by other means.

## Interpolation demo

```{code-cell}
using Distances

x = [0,0.1,0.31,0.48,0.66,0.87,1]
R = pairwise(Euclidean(),x)
```

```{code-cell}
φ = r -> sqrt(1+r^2)
A = φ.(R)
```

```{code-cell}
f = x -> exp(sin(2x))
c = A \ f.(x)
```

```{code-cell}
using LinearAlgebra
cond(A)
```

```{code-cell}
using Plots
plotly()
default(label="",linewidth=2)
plot(f,0,1)
scatter!(x,f.(x))

u = t -> sum(c[j]*φ(norm(t-x[j])) for j in eachindex(x))
plot!(u,0,1)
```

```{code-cell}
x = range(-1,1,21)
R = pairwise(Euclidean(),x)
logε = range(-2,3,100)
xx = range(-1,1,1800)

κ = []
err = []
for logε in logε
    ε = 10^logε
    A = φ.(ε*R)
    push!(κ,cond(A))
    c = A\f.(x)
    u = t -> sum(c[j]*φ(ε*norm(t-x[j])) for j in eachindex(x))
    push!(err,norm(f.(xx)-u.(xx),Inf))
end
plot(10 .^logε,κ,xaxis=("ε",:log10),yaxis=(:log10),label="cond")
plot!(10 .^logε,err,label="err",title="Shape parameter")
```

```{code-cell}
xx = range(-1,1,1800)
n = 10:10:320
κ = []
err = []

for n in n
    x = range(-1,1,n+1)
    R = pairwise(Euclidean(),x)
    ε = 10
    A = φ.(ε*R)
    push!(κ,cond(A))
    c = A\f.(x)
    u = t -> sum(c[j]*φ(ε*norm(t-x[j])) for j in eachindex(x))
    push!(err,norm(f.(xx)-u.(xx),Inf))
end
plot(n,1.0./κ,xaxis=("N"),yaxis=(:log10),label="1/cond")
plot!(n,err,label="err",title="Nonstationary convergence")
```

```{code-cell}
xx = range(-1,1,1800)
n = 10:10:320
κ = []
err = []

for n in n
    x = range(-1,1,n+1)
    R = pairwise(Euclidean(),x)
    ε = 0.5/(2/n)
    A = φ.(ε*R)
    push!(κ,cond(A))
    c = A\f.(x)
    u = t -> sum(c[j]*φ(ε*norm(t-x[j])) for j in eachindex(x))
    push!(err,norm(f.(xx)-u.(xx),Inf))
end
plot(n,1.0./κ,xaxis=("N"),yaxis=(:log10),label="1/cond")
plot!(n,err,label="err",title="Stationary nonconvergence")
```

## PDE collocation

In the context of PDEs, there are many ways to deploy RBFs. The natural starting point is collocation. To solve the linear problem

$$
Lu &= f, \qquad x \in \Omega, \\ 
Bu &= g, \qquad x \in \partial\Omega, 
$$

we can replace interpolation conditions by collocation conditions. Suppose we have centers $x_1,\ldots,x_N$ in the interior $\Omega$ and $x_{N+1},\ldots,x_{N+\nu}$ on the boundary $\partial \Omega$. Then

$$
f(x_i) = \sum_{n=1}^N c_n (L \varphi)\bigl(\| x_i - x_n \|\bigr) + \sum_{m=1}^M a_m (L\psi_m)(x_i), \quad i = 1,\ldots,N,
$$

and

$$
g(x_i) = \sum_{n=1}^N c_n (B \varphi)\bigl(\| x_i - x_n \|\bigr) + \sum_{m=1}^M a_m (B\psi_m)(x_i), \quad i = N+1,\ldots,N+\nu. 
$$

This is still a linear system to solve for the coefficients.

```{code-cell}
t = range(-1.1,1.1,64)
isinside(x,y) = x^2 + 3y^2 < 0.97
XI = [ [x,y] for x in t, y in t if isinside(x,y)]
scatter([x[1] for x in XI],[x[2] for x in XI],m=2,msw=0)
```

```{code-cell}
XB = [ [cos(θ),sin(θ)/sqrt(3)] for θ in range(0,2π,150) ]
scatter!([x[1] for x in XB],[x[2] for x in XB],m=2,msw=0)
```

```{code-cell}
φ = r -> r^5
Lφ = r -> 25r^3
x = [XI;XB]
R = pairwise(Euclidean(),x)
Ni,Nb = length(XI),length(XB)
N = Ni+Nb
A = Lφ.(R[1:Ni,:])
B = φ.(R[Ni+1:N,:])
f = [-ones(Ni);[x[1]^2+x[2] for x in XB]]
c = [A;B] \ f
```

```{code-cell}
u = t -> isinside(t...) ? sum(c[j]*φ(norm(t-x[j])) for j in eachindex(x)) : NaN
xx = range(-1,1,130)
surface(xx,xx,[u([x,y]) for x in xx, y in xx])
```

Again, but with constant and linear terms added:

```{code-cell}
P = [ones(Ni+Nb) [x[1] for x in x] [x[2] for x in x]]
c = [ [A;B] P; P' zeros(3,3) ] \ [f; zeros(3)];
```

```{code-cell}
u = function(t)
    if isinside(t...)
        u1 = sum(c[j]*φ(norm(t-x[j])) for j in 1:N) 
        u2 = c[N+1] + c[N+2]*t[1] + c[N+3]*t[2]
        return u1 + u2
    else
        return NaN
    end
end
xx = range(-1,1,130)
surface(xx,xx,[u([x,y]) for x in xx, y in xx])
```
