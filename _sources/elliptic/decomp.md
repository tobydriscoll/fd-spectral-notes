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

```{code-cell}
using LinearAlgebra
⊗ = kron
include("/Users/driscoll/817/notes/elliptic/diffmats.jl")

function isboundary(xy)
    x,y = xy
    return (x==-1) | (x==1) | (y==-1) | (y==1) | 
        ((x==0) & (y>=0)) | ((y==0) & (x>=0))
end

function isinterface(xy)
    return ((xy[1]==0) & (-1<xy[2]<0))
end
```

```{code-cell}
function domains(n)
    R₁ = (nx=n,xspan=(-1,0),ny=2n,yspan=(-1,1),normal=1.)
    R₂ = (nx=n,xspan=(0,1),ny=n,yspan=(-1,0),normal=-1.) 

    region = []
    for R in (R₁,R₂)
        nx,ny = R.nx,R.ny
        x,Dx,Dxx = diffmats(R.nx,R.xspan...)
        y,Dy,Dyy = diffmats(R.ny,R.yspan...)
        N = (nx+1)*(ny+1)
        Δ = I(ny+1)⊗Dxx + Dyy⊗I(nx+1)
        grid = vec([[x,y] for x in x, y in y])
        onbdy = isboundary.(grid)
        oniface = isinterface.(grid)
        interior = @. !onbdy & !oniface
        Dnorm = R.normal*(I(ny+1)⊗Dx)[oniface,:]
        
        push!(region,(;x,y,nx,ny,N,Δ,grid,onbdy,oniface,interior,Dnorm))
    end

    offset = [0,region[1].N]
    idx = []
    for (k,r) in zip(offset,region)
        all = k.+(1:r.N)
        iface = k.+findall(r.oniface)
        bdy = k.+findall(r.onbdy)
        push!(idx,(;all,iface,bdy))
    end
    
    return region,idx
end;
```

```{code-cell}
using Plots
region,idx = domains(10)
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
f(x) = x[1]+2  # forcing function
region,idx = domains(40)

N = sum(r.N for r in region)
A = sparse(zeros(N,N))
b = zeros(N)
for (i,r) in zip(idx,region)
    A[i.all,i.all] = r.Δ
    b[i.all] = f.(r.grid)
    A[i.bdy,:] .= I(N)[i.bdy,:]
    b[i.bdy] .= 0
end
default(size=(300,300))
spy(A,color=:blues)
```

```{code-cell}
Ni = length(idx[1].iface)
A[idx[1].iface,:] .= 0
A[idx[1].iface,idx[1].iface] = spdiagm(ones(Ni))
A[idx[1].iface,idx[2].iface] = -spdiagm(ones(Ni))
b[idx[1].iface] .= 0
spy(A,color=:blues)
```

```{code-cell}
A[idx[2].iface,:] .= 0
A[idx[2].iface,idx[1].all] = region[1].Dnorm
A[idx[2].iface,idx[2].all] = region[2].Dnorm
b[idx[2].iface] .= 0
spy(A,color=:blues)
```

```{code-cell}
u = A\b;
```

```{code-cell}
plt = plot([-1,-1,1,1,0,0,-1],[1,-1,-1,0,0,1,1],l=(2,:black),label="",aspect_ratio=1)
for (i,r) in zip(idx,region)
    contour!(r.x,r.y,u[i.all],levels=-.25:0.01:0)
end
plt
```

```{code-cell}
⊗ = kron

function isboundary(xy)
    x,y = xy
    return (x==-1) | (x==1) | (y==-1) | (y==1) | 
        ((x==0) & (y>=0)) | ((y==0) & (x>=0))
end

function isinside(xy,xspan,yspan)
    x,y = xy
    return (xspan[1] < x < xspan[2]) && (yspan[1] < y < yspan[2])
end

function isinterface(xy)
    return ((xy[1]==0) & (-1<xy[2]<0))
end

function domains(n)
    R = [ 
    (nx=n,xspan=(-1,0),ny=2n,yspan=(-1,1)),
    (nx=n,xspan=(-0.23,1),ny=n,yspan=(-1,0))
    ]

    region = []
    for i in 1:2
        nx,ny = R[i].nx,R[i].ny
        x,Dx,Dxx = diffmats(R[i].nx,R[i].xspan...)
        y,Dy,Dyy = diffmats(R[i].ny,R[i].yspan...)
        N = (nx+1)*(ny+1)
        Δ = I(ny+1)⊗Dxx + Dyy⊗I(nx+1)
        grid = [[x,y] for x in x, y in y]
        interior = falses(nx+1,ny+1)
        interior[2:nx,2:ny] .= true
        oniface = .!interior
        otherR = R[3-i]
        for idx in findall(oniface)
            if !isinside(grid[idx],otherR.xspan,otherR.yspan)
                oniface[idx] = false
            end
        end
        onbdy = isboundary.(grid)
        push!(region,(;x,y,nx,ny,N,Δ,grid,onbdy,oniface,interior))
    end
    return region
end
```

```{code-cell}
using Plots
region = domains(12)
fig = plot(aspect_ratio=1,m=3,leg=false)
for (r,sym) in zip(region,[:x,:+])
    for s in [r.interior,r.onbdy,r.oniface]
        x,y = [p[1] for p in r.grid[s]],[p[2] for p in r.grid[s]]
        scatter!(x,y,m=sym,msw=2)
    end
end
fig
```

```{code-cell}
using BlockArrays,Dierckx

function schwarz(u,f,region)
    for (i,R) in enumerate(region)    
        A = R.Δ
        b = f.(vec(R.grid))

        # True boundary conditions (Dirichlet)
        onbdy = vec(R.onbdy)
        for idx in findall(onbdy)
            b[idx] = 0
            A[idx,:] .= 0
            A[idx,idx] = 1
        end

        # Interface conditions
        other = 3-i
        uu = reshape(u[Block(other)],region[other].nx+1,region[other].ny+1)
        s = Spline2D(region[other].x,region[other].y,uu,kx=1,ky=1)
        oniface = vec(R.oniface)
        for idx in findall(oniface)
            b[idx] = s(R.grid[idx]...)
            A[idx,:] .= 0
            A[idx,idx] = 1
        end
        
        u[Block(i)] .= A\b
    end
    return u
end
```

```{code-cell}
region = domains(40)
f(x) = x[1]+2  # forcing function
N = [r.N for r in region]
u = BlockVector(zeros(sum(N)),N)

anim = @animate for i in 1:10
    global u
    plt = plot([-1,-1,1,1,0,0,-1],[1,-1,-1,0,0,1,1],l=(2,:black),label="",
        aspect_ratio=1,dpi=100)
    for (i,R) in enumerate(region)
        U = reshape(u[Block(i)],R.nx+1,R.ny+1)
        contour!(R.x,R.y,U',levels=-.25:0.01:0)
    end
    u = schwarz(u,f,region)
    plt
end

mp4(anim,"schwarz.mp4",fps=1)
```

```{code-cell}

```
