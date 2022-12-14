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

# Finite differences

We begin with the matter of accurately representing differentiation operators on finite collections of values. For example, suppose we are given $h>0$ and the values $u_1=f(h)$ and $u_0=f(0)$. It's clear that we can approximate $f'(0)$ using the **forward-difference** quotient

$$
f'(0) \approx \frac{f(h)-f(0)}{h} = \frac{u_1-u_0}{h}. 
$$

If we have access to $u_{-1}=f(-h)$, we might instead use the **backward difference**

$$
f'(0) \approx \frac{f(0)-f(-h)}{h} = \frac{u_0-u_{-1}}{h},
$$

or the average of the two, which is the **centered difference**

$$
f'(0) \approx \frac{f(h)-f(-h)}{2h} = \frac{u_1-u_{-1}}{2h}.
$$

Note that these formulas, like the exact differentiation operator, are essentially translation-invariant. If we are given the values $f(10+h)$ and $f(10-h)$, for instance, then a centered difference of them would give an approximation to $f'(10)$. 

Immediately we see that we're going to have lots of options. While that can be great, it requires us to analyze the strengths and weaknesses of each candidate in the context that it will be applied.
