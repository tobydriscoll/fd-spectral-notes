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

# Chebyshev for BVPs

For the most part, we can drop in a Chebyshev differentiation matrix where we previously used finite differences, but there are some important differences to keep in mind.

## 1D BVP

Consider 

$$
\partial_{xx} u = e^{4x}, \qquad u(\pm 1) = 0. 
$$

Unlike FD methods, a reasonable way to get the matrix $\bfD_{xx}$ is by squaring $\bfD_x$. (We drop the $N$ subscript for notational sanity.) Mathematically, both reduce to evaluation of the 2nd derivative of the global polynomial interpolant. For the homogeneous Dirichlet conditions in this example, the numerical solution satisfies $v_0=v_N=0$, so we can remove both them and the columns of $\bfD_{xx}$ that they multiply from the linear system $\bfD_{xx} = \exp(4\bfx)$. 

The solution of the linear system produces the grid function $v$ at Chebyshev points. Unlike FD methods, however, these values also imply a global interpolant over the solution domain. In fact, it would be a crime to use linear interpolation or a cubic spline for off-grid points, as those methods would erase the advantage of spectral accuracy (though you would be unlikely to notice much on a plot of the solution for any smooth method). We use the `polyinterp` function defined earlier to evaluate the solution anywhere; in principle, we should specialize it to Chebyshev points, for which the barycentric weights are known in closed form, but the performance issue is not meaningful to us for these demonstrations.

```{code-cell} julia
using Sugar, SpectralMethodsTrefethen
Sugar.get_source(first(methods(p13))) |> last |> print
p13()
```

For the nonlinear variant 

$$
\partial_{xx} u = e^{u}, \qquad u(\pm 1) = 0, 
$$

we again derive a nonlinear algebraic system by discretization. Here is a code that uses a fixed-point iteration to approximately solve the nonlinear system:

```{code-cell} julia
Sugar.get_source(first(methods(p14))) |> last |> print
p14()
```

Recall that there are two senses of convergence here: convergence of the iteration as a solution of the discrete equations, and convergence of the discretization to the underlying solution function.

We can also solve the Laplacian eigenvalue problem 

$$
\partial_{xx} u = \lambda {u}, \qquad u(\pm 1) = 0.
$$

We have the exact solutions 

$$
\lambda_n = -\frac{n^2\pi^2}{4}, \qquad u_n(x) = \sin[n\pi(x+1)/2], \qquad n = 1,2,\dots. 
$$

```{code-cell} julia
Sugar.get_source(first(methods(p15))) |> last |> print
p15()
```

Observe above that the results become increasingly inaccurate as the eigenfunction wavenumber increases. The resolution of a method is often expressed in terms of **points per wavelength (PPW)**, which is the wavelength divided by the grid spacing. As a reference, in Fourier methods the highest wavenumber that can be resolved is $k=N/2$, and the PPW in this case is

$$
\frac{4\pi/N}{2\pi/N} = 2. 
$$

Thus, 2 PPW is the bare minimum for a Fourier spectral method. 

For a Chebyshev method, the points are coarsest in the center of the grid, where the node density is $1/\pi$. This compares to an equispaced node density (on $[-1,1]$) of $1/2$, so we lose a factor of $\pi/2$ in resolution at the center by comparison. It's therefore reasonable to state that Chebyshev methods have a $\pi$ PPW minimum resolution requirement.

In the example above, the wavelength of the $n$th mode is $4/n$. Compared to $h=\pi/N$ at the center, the effective PPW is therefore $4N/(n\pi)$ at mode $n$. The figure shows that about 7 digits are correct at PPW 3.1, but the accuracy falls off quickly beyond that.

## 2D Poisson and Helmholtz

For problems over a rectangle, we can use Kronecker products on a tensor product of Chebyshev grids. The resulting discrete approximation to, say, the Laplacian operator is less sparse than we see with finite differences, the idea is that a much smaller matrix will suffice for equivalent accuracy. If the matrix is truly too large to handle with dense linear algebra, then one can turn to iterative methods with a fast evaluation alternative to be presented in the next section.

One new wrinkle in 2D is that the interpolant must also be evaluated in a tensor-product fashion. For example, suppose a grid function has $U_{ij}$ given at all $(x_i,y_j)$ for independent Chebyshev grids in $x$ and $y$. To evaluate the interpolant at $(\xi,\eta)$, we first evaluate 1D interpolants at $(\xi,y_j)$ for all $j$, and then do one more 1D evaluation at $(\xi,\eta)$. This can be visualized as collapsing each column of data down to the point $\xi$, then collapsing the remaining row to a single point. In practice we can do this reasonably efficiently on a grid of $(\xi,\eta)$ values, and we implement that as `gridinterp`.

```{code-cell} julia
Sugar.get_source(first(methods(gridinterp))) |> last |> print
```

Here is a solution of $\Delta u = 10\sin(8x(y-1))$ over $[-1,1]^2$ with homogeneous Dirichlet conditions:

```{code-cell} julia
Sugar.get_source(first(methods(p16))) |> last |> print
fig1,fig2 = p16();
fig1
```

```{code-cell} julia
fig2
```

An important variation on the Poisson equation is the **Helmholtz equation**, which plays a major role in wave propagation:

$$
\Delta u + k^2 u = f(x,y),
$$

where $k$ is a real parameter. Here is an example solution:

```{code-cell} julia
Sugar.get_source(first(methods(p17))) |> last |> print
p17()
```
