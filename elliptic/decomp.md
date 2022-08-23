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

# Domain decomposition

Rectangles are great and all, but we might like to go beyond them. One attractive technique that generalizes the geometry somewhat is **domain decomposition**. It can also break a large problem into smaller pieces that are more manageable and perhaps solvable in parallel.

There are two major types of domain decomposition.

## Nonoverlapping

In a nonoverlapping DD method, domain $\Omega$ is broken into $\Omega_i$ that intersect only at lower-dimensional interfaces. 

For simplicity, consider a domain with two subregions $\Omega_1$ and $\Omega_2$, with $\Gamma_i = \partial \Omega_i \cap \partial \Omega$ and $\Gamma_{12} = \partial \Omega_1 \cap \partial \Omega_2$. You can imagine that both $\Omega_i$ are rectangles in practice, but this is not really important in principle. It can be shown that the solution of 

$$
\Delta u &= 0, \quad \text{in } \Omega, \\ 
Bu &= 0, \quad \text{on } \partial \Omega, 
$$

where $B$ is a linear operator of differential order 0 or 1, is equivalent to the solution of the subproblems

$$
\Delta u_i &= 0, \quad \text{in } \Omega_i, \\ 
Bu_i &= 0, \quad \text{on } \Gamma_i,  
$$

coupled by the **interface conditions** 

$$
u_1 = u_2, \quad \pp{u_1}{n} = -\pp{u_2}{n}, \qquad \text{on }  \Gamma_{12}.
$$

Now let us discretize the subregions. The equations for the non-interface nodes of $\Omega_1$ have the form

$$
\begin{bmatrix}
  \bfA_{1o} & \bfA_{1i}
\end{bmatrix}
\begin{bmatrix}
  bfu_{1o} \\ \bfu_{1i}
\end{bmatrix}
= \bfb_1, 
$$

where we have partitioned into ordinary/interface nodes. Doing similarly for $\Omega_2$ and adding the interface conditions gives

$$
\begin{bmatrix}
  \bfA_{1o} & & \bfA_{1i} & \\ 
  & \bfA_{2o} & & \bfA_{2i} \\ 
  & & & \mathbf{I} & -\mathbf{I} \\ 
  & & \bfD_{n1} & \bfD_{n2}
\end{bmatrix}
\begin{bmatrix}
  bfu_{1o} \\ \bfu_{2o} \\ \bfu_{1i} \\ \bfu_{2i}
\end{bmatrix}
= 
\begin{bmatrix}
  \bfb_1 \\ \bfb_2 \\ \bfzero \\ \bfzero 
\end{bmatrix}. 
$$


```{code-cell}
using LinearAlgebra
⊗ = kron
include("diffmats.jl")

n = 8
R₁ = (x=range(-1,0,n+1),y=range(-1,1,2n+1))
R₂ = (x=range(0,1,n+1),y=range(-1,0,n+1)) 
R₁ = (nx=n,xspan=(-1,0),ny=2n,yspan=(-1,1))
R₂ = (nx=n,xspan=(0,1),ny=n,yspan=(-1,0)) 

f(x) = x[1]+2

function isboundary(xy)
    x,y = xy
    return (x==-1) | (x==1) | (y==-1) | (y==1) | 
        ((x==0) & (y>=0)) | ((y==0) & (x>=0))
end

function isinterface(xy)
    return ((xy[1]==0) & (-1<xy[2]<0))
end

region = []
for R in (R₁,R₂)
    # x,y = R.x,R.y
    # nx,ny = length(x),length(y)
    # N = nx*ny
    nx,ny = R.nx,R.ny
    x,Dx,Dxx = diffmats(R.nx,R.xspan...)
    y,Dy,Dyy = diffmats(R.ny,R.yspan...)
    N = (nx+1)*(ny+1)
    Δ = I(ny+1)⊗Dxx + Dyy'⊗I(nx+1)
    grid = vec([[x,y] for x in x, y in y])
    onbdy = isboundary.(grid)
    oniface = isinterface.(grid)
    interior = @. !onbdy & !oniface
    Dnorm = (I(ny+1)⊗Dx)[oniface,:]
    push!(region,(;x,y,nx,ny,N,Δ,grid,onbdy,oniface,interior,Dnorm))
end
```

```{code-cell}
points = [region[1].grid;region[2].grid]
offset = [0,region[1].N]
idx = []
for (k,r) in zip(offset,region)
    all = k.+(1:r.N)
    iface = k.+findall(r.oniface)
    bdy = k.+findall(r.onbdy)
    push!(idx,(;all,iface,bdy))
end
```

```{code-cell}
using Plots
fig = plot(aspect_ratio=1,m=3,leg=false)
for (i,r,sym) in zip(idx,region,[:x,:+])
    for s in [r.interior,r.onbdy,r.oniface]
        x,y = [p[1] for p in r.grid[s]],[p[2] for p in r.grid[s]]
        scatter!(x,y,m=sym,msw=2)
    end
end
fig
```

```{code-cell}
N = sum(r.N for r in region)
A = sparse(zeros(N,N))
b = zeros(N)
for (i,r) in zip(idx,region)
    A[i.all,i.all] = r.Δ
    b[i.all] = f.(r.grid)
    A[i.bdy,:] .= 0
    A[i.bdy,i.bdy] .= diagm(ones(length(i.bdy)))
    b[i.bdy] .= 0
end
```

```{code-cell}
spy(A,color=:redsblues)
```

```{code-cell}
Ni = length(idx[1].iface)
A[idx[1].iface,:] .= 0
A[idx[1].iface,idx[1].iface] = spdiagm(ones(Ni))
A[idx[1].iface,idx[2].iface] = -spdiagm(ones(Ni))
b[idx[1].iface] .= 0

A[idx[2].iface,:] .= 0
A[idx[2].iface,idx[1].all] = region[1].Dnorm
A[idx[2].iface,idx[2].all] = -region[2].Dnorm
b[idx[2].iface] .= 0
spy(A,color=:redsblues)
```

```{code-cell}
u = A\b;
```

```{code-cell}
gr()
plt = plot(aspect_ratio=1)
for (i,r) in zip(idx,region)
    contour!(r.x,r.y,u[i.all],levels=-.25:0.01:0)
end
plt
```

We could again use the Schur complementation technique to remove the interface unknowns from the linear system. 

## Overlapping

A more versatile technique is to use overlapping subdomains. 
