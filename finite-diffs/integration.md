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

# Indefinite integration

The simplest possible context for our first finite-difference method is the BVP

$$
u'(x) = g(x), \quad u(a)=0, \quad a < x < b. 
$$

The solution of this problem is just an indefinite integral of $g$. Suppose we discretize the domain $[a,b]$ by choosing the nodes

$$
x_i = a + ih, \quad h = \frac{b-a}{n}, \quad i=0,\ldots,n,
$$

for a positive integer $n$. If we want to write a discrete analog of the ODE at each node, then at the first node $x_0$ we really only have one of the three options above available, the forward difference. Likewise, at $x_n$ we can only use the backward difference. At the interior nodes, suppose that we use a centered difference. This gives us the discrete equations

$$
\frac{u_1-u_0}{h} = g(x_0), \, \frac{u_2-u_0}{2h} = g(x_1), \, \cdots \, \frac{u_n-u_{n-2}}{2h} = g(x_{n-1}),  \, \frac{u_n-u_{n-1}}{h} = g(x_{n}). 
$$

It's much saner to express these using linear algebra:

$$
\frac{1}{2h}
\begin{bmatrix} 
-2 & 2 & & & \\ 
-1 & 0 & 1 & & \\ 
& \ddots & \ddots & \ddots  & \\ 
& & -1 & 0 & 1 \\ 
& & & -2 & 2 
\end{bmatrix} 
\begin{bmatrix} u_0 \\ u_1 \\ \vdots \\ u_{n-1} \\ u_n   \end{bmatrix}
= 
\begin{bmatrix} g(x_0) \\ g(x_1) \\ \vdots \\ g(x_{n-1}) \\ g(x_n)   \end{bmatrix}. 
$$

We will call the matrix in this system a **differentiation matrix**. It maps a vector of function values to a vector of (approximate) values of its derivative.

```{code-cell}
using LinearAlgebra
function diffmat1(x)
    # assumes evenly spaced nodes
    h = x[2]-x[1]
    m = length(x)
    Dx = 1/2h*diagm(-1=>[-1;-2*ones(m-2)],0=>[-2;zeros(m-2);2],1=>[2;ones(m-2)])
end

a,b = 0,1
g = x->cos(x)

n = 8
h = (b-a)/n
x = [a + i*h for i in 0:n]
A = diffmat1(x)
```

```{code-cell}
b = g.(x)
u = A\b
```

Wait, what??

Sometimes when the discrete analog of the problem fails spectacularly, it's actually reflecting an important property of the original problem. Here, we've forgotten to impose the boundary/initial condition $u(a)=0$, and that makes the problem ill-posed. In fact, we know that the integration problem has a 1-dimensional nonuniqueness, and so does the matrix problem here:

```{code-cell}
U,σ,V = svd(A)
σ
```

The null space even consists of a constant solution:

```{code-cell}
V[:,end]
```

Let's impose the boundary condition now. In order to keep this a square linear system, we need to replace a row, not add one; the natural choice is the first row.

```{code-cell}
A[1,1:2] = [1,0];  b[1] = 0;
u = A\b 
```

Compare to the exact solution, $u(x)=\sin(x)$:

```{code-cell}
[u sin.(x)]
```

While that's plausible, it's not conclusively correct. Let's do a convergence study.

```{code-cell}
function fdintegrate(g,a,b,n)
    h = (b-a)/n
    x = [a + i*h for i in 0:n]
    A = diffmat1(g,a,b,n)
    b = g.(x)

    A[1,1:2] = [1,0];  b[1] = 0;
    
    u = A\b 
    return x,u
end

n = [2^k for k in 1:10]
err = []
for n in n
    x,u = fdintegrate(cos,0,1,n)
    û = sin.(x)
    push!(err,norm(u-û)/norm(û))
end

using Plots
plot(n,err,m=4,xaxis=:log10,yaxis=:log10)
order1 = @. (n/n[1])^(-2)
plot!(n,order1,color=:black,l=:dash)
```

The straight-line convergence on log-log scales means **algebraic convergence** of the rate $C n^{-p}$ for a positive **order** $p$. The dashed black line shows $p=2$. How do we explain this convergence order? Stay tuned.
