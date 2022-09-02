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

# Bootstrapping methods

*I made this term up for this context. Nobody else uses it.*

There are many interesting and useful ways to merge analytical understanding of numerical solutions with numerical practice in order to boost accuracy without using a fundamentally new discretization method. Here are introductions to two of them.

## Extrapolation

Suppose we have numerical solution $\bfu$ on a grid with spacing $h$, approximating the exact solution $\hat{u}(x)$, and that 

$$
\hat{u}(x_j) - u_j = c_2 h^2 + c_4 h^4 + \cdots. 
$$

Here, $c_2,c_4,\dots$ are constant with respect to $h$, though they usually do depend on the grid location and the exact solution $\hat{u}$. Crucially, *we don't need to know the values of the constants*, just that the form of the expansion is valid.

Now suppose $\bfv$ is another discrete solution on a grid with spacing $h/2$. Then 

$$
\hat{u}(x_j) - v_j = \frac{c_2}{4} h^2 + \frac{c_4}{16} h^4 + \cdots. 
$$

It follows that

$$
\frac{1}{3} \left( 4v_{2j} - u_j  \right) = \hat{u}(x_j) + \tilde{c}_4 h^4 + \cdots . 
$$

That is, we can construct a 4th-order solution out of two 2nd-order solutions. If a third solution is available, then two 4th-order solutions can be combined to get 6th-order, etc. This is the method of **extrapolation**.

It's an appealing idea, but not always easy to pull off in practice. Boundary conditions can interfere with the neat error expansion, and extrapolation is sensitive to perturbations if carried out to a large extent. 

## Deferred correction

Consider a discretized linear problem $\bfA \bfu = \bff$. Recall that we define local truncation error as 

$$
\mathbf{t} = \bfA \bigl[ \hat{u}(x_j) \bigr]_j - \bff, 
$$

using the exact solution $\hat{u}$. Defining $\bfe = [ \hat{u}(x_j) ] - \bfu$ as the error vector, it follows that 

$$
\bfA \bfe = \bfA \bigl[ \hat{u}(x_j) \bigr]_j - \bfA \bfu = \mathbf{t}. 
$$

Therefore, if we can estimate $\mathbf{t}$ to sufficient accuracy, we can solve another linear system to get a correction to the solution. This is the method of **deferred correction** (although that term is applied to a wide variety of methods and problem types).

For example, consider the *Airy equation* $u''-xu=0$ discretized by 2nd-order differences. Away from the boundaries, 

$$
t_j &= \hat{u}''(x_j) + \frac{h^2}{12} \hat{u}''''(x_j) + O(h^4)  - x_j \hat{u}(x_j)  \\
&= \frac{h^2}{12}  \left[ (\hat{u}(x))''\right]''_{x_j} + O(h^4)  \\ 
&= \frac{h^2}{12} \left[x u(x)\right]''_{x_j} + O(h^4). 
$$

We can approximate the bracketed term over the grid as 

$$
\bfD_{xx} (\diag{\bfx} \hat{\bfu} ) = \bfD_{xx} (\diag{\bfx} \bfu ) + O(h^2). 
$$

Thus, the leading term of $\mathbf{t}$ is computable, and we can find an approximate correction to $\bfu$.

