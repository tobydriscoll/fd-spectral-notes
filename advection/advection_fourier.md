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

# von Neumann analysis

A von Neumann analysis is one of the easier ways to draw quantitative conclusions about stability and accuracy. Here we look at 

$$
\partial_t u + c \partial_x u = 0
$$

on the entire real line. 

## Centered in space

We can proceed in semidiscrete form. Letting

$$
u(x,t) = v(t) e^{i\xi x},
$$

we get

$$
v'(t) &= -c \frac{e^{i\xi h}-e^{-i\xi h}}{2h} v(t)\\ 
 &= -\frac{ic}{h} v(t) \sin(\xi h)\\ 
 & = \lambda(\xi) v(t).
$$

This is the absolute stability model problem. (Not coincidentally, we see here behavior similar to the eigenvalues found in the periodic case, ranging between $\pm ic/h$ on the imaginary axis.) Stability analysis is identical to checking that $\tau \lambda(\xi)$ lies within the stability region of the method used to discretize time.

There is, however, a little more information to be found in the details. Suppose we choose Euler in time and look for a solution in the form

$$
v(t_j) = g(\xi)^j,
$$

where $g$ is known as the **amplification factor**. We get $g^{j+1} = g^j + \tau \lambda g^j$, or 

$$
g(\xi) = 1 + \tau \lambda(\xi).
$$

Since $\lambda$ is purely imaginary, we have $|g| > 1$ except when $\lambda=0$. Furthermore, $|g|$ is maximized when $|\lambda|$ is maximized, which happens when $|\sin(\xi h)|=1$, or $\xi h = \pm \pi/2$. This is the mode that will blow up most rapidly.

```{code-cell}
include("diffmats.jl")
```

```{code-cell}
using OrdinaryDiffEq

n = 100
c = -1
u₀ = x -> exp(2*sin(2π*x))

x,Dx,_ = diffmats(n,0,1,periodic=true)
advect(u,c,t) = -c*(Dx*u)

ivp = ODEProblem(advect,u₀.(x),(0.,10.),c)
sol = solve(ivp,Euler(),dt=1/1000);
```

```{code-cell}
using Plots,PyFormattedStrings
anim = @animate for t in range(0,8,201)
    plot(x,sol(t),label=f"t={t:.1f}",m=2,
        xaxis=("x"),yaxis=("u(x,t)",[-3,9]),dpi=140)
end
mp4(anim,"advect1.mp4")
```

Note the onset of an exponential instability takes a while. Here is a snapshot shortly after it becomes obvious:

```{code-cell}
plot(x,sol(7.4),m=2)
```

The pattern here has oscillations with just one grid point in-between successive extremes. This is like $-1,0,1,0,-1,0,\ldots$, as opposed to the sawtooth $-1,1,-1,1,\ldots$, so we are at half the max wavenumber, or $\xi = \pi/2h$. 

Suppose instead we use the trapezoid formula AM2. Then

$$
g = 1 + \tfrac{1}{2}\tau (1+g) \lambda, 
$$

or 

$$
g(\xi) = \frac{2+\tau\lambda}{2-\tau\lambda}. 
$$

The fact that $\lambda$ is imaginary now implies $|g|=1$ at all wavenumbers. This is ideal from the perspective that the exact solutions of the problem do not decay or grow.


## Upwind

For an upwind discretization with $c>0$, we must use backward differences, leading to 

$$
v'(t) &= -c \frac{1-e^{-i\xi h}}{h} v(t)\\ 
 & = \lambda(\xi) v(t).
$$

In the complex plane, $-e^{-i\xi h}$ covers the unit circle as $\xi h$ ranges over $[-\pi,\pi)$. This circle is shifted to have center at $z=1$ and then scaled by $-c/h$ to get $\lambda$. This result implies that content at all wavenumbers except $\xi=0$ is damped in time. The damping is strongest at the sawtooth wavenumber $\xi = \pm \pi/h$. The damping is a property of the upwind spatial discretization, independently of how the time integration is carried out.

If we use Euler in time, then absolute stability will follow if $-2c\tau /h \ge -2$, or $\tau \le h/c$. By contrast, both trapezoid and backward Euler are unconditionally stable.  

```{code-cell}

```

```{code-cell}

```

```{code-cell}

```
