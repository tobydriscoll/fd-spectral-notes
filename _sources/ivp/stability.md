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

# Absolute stability

Like FD formulas, an IVP formula has a local truncation error and order of accuracy. These measure how well the formula approximates the ODE, but it's not immediate that the formula's solution converges to the true one. We also need **stability**, which (again) means that the solution remains bounded as the step size approaches zero. The celebrated **Dahlquist Equivalence Theorem** states that accuracy and stability together imply convergence (at the same order of accuracy as the LTE).

However, there is another sense of stability which is just as important in practice, based on the model problem

:::{math}
:label: absstabmodel
y' = \lambda y, \quad y(0)=1.
:::

::::{prf:definition} Absolute stability
Let $\lambda$ be a complex number, and let $y_0,y_1,y_2,\ldots,y_n$ be the numerical solution at times $0,\tau,2\tau,\ldots,n\tau$ of {eq}`absstabmodel` using a Runge–Kutta or multistep method with fixed step size $\tau$. Then the method is said to be **absolutely stable** at $\zeta = \tau\lambda$ if $|y_n|$ is bounded above as $n\to\infty$. 
::::

::::{prf:observation}
Solutions of {eq}`absstabmodel` are bounded as $t\to\infty$ if and only if $\alpha = \operatorname{Re} \lambda \le 0$. 
::::

The fact that absolute stability depends only on the product $\zeta = \tau\lambda$, and not independently on the individual factors, is a result of how the IVP solvers are defined, as we will see below. Since $\lambda$ has units of inverse time, $\zeta$ is dimensionless.



## Step size selection

Modern solvers have a means of not just producing a numerical solution but also estimating the error in it. Since we are usually more interested in controlling the error than selecting a particular step size, the solver predicts a tentative solution, rejecting or accepting it based on whether it seems to be as accurate as the user would like, and then adjusting the step size to run as closely as possible to the error tolerance without incurring a needlessly long computation time. Ideally, the step size shrinks and grows along with the time scale of changes in the solution.

```{code-cell}
t = sol.t
Δt = [t[k+1]-t[k] for k in 1:length(t)-1]
plot(layout=(2,1),link=:x)
plot!(sol,subplot=1)
plot!(t[1:end-1],Δt,subplot=2)
```

Convergence of an IVP solver requires stability, in the form of a bounded solution as the step size $\tau\to 0$. However, there is another important sense of stability for a *fixed* step size. 