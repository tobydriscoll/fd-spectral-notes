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

# Iterative linear algebra

## Krylov methods

```{code-cell}
include("/Users/driscoll/817/notes/elliptic/diffmats.jl")

m,n = 3,5
x,Dx,Dxx = diffmats(m,0,1)
y,Dy,Dyy = diffmats(n,-1,1)

function laplacian(v)
    V = reshape(v,m+1,n+1)
    AV = Dxx*V + V*Dyy'
    return vec(AV)
end

using LinearMaps
A = LinearMap(laplacian,((m+1)*(n+1)));
```

```{code-cell}
A
```

```{code-cell}
A*ones(24)
```

```{code-cell}
Matrix(A)
```

```{code-cell}
using Krylov

# Solution of Poisson problem with u=g on boundary
function iterative(m,n,xspan,yspan,f,g)
    x,Dx,Dxx = diffmats(m,xspan...)
    y,Dy,Dyy = diffmats(n,yspan...)
    grid = [(x,y) for x in x, y in y]

    # Identify boundary locations.
    isboundary = trues(m+1,n+1)
    isboundary[2:m,2:n] .= false
    idx = vec(isboundary);

    # forcing function / boundary values vector
    b = vec( f.(grid) )
    b[idx] = g.(grid[idx]);   # assigned values
    
    # Apply Laplacian operator with Dirichlet condition.
    function laplacian(v)
        V = reshape(v,m+1,n+1)
        AV = Dxx*V + V*Dyy'
        AV[idx] .= V[idx]   # Dirichlet condition
        return vec(AV)
    end

    A = LinearMap(laplacian,(m+1)*(n+1))
    u,stats = gmres(A,b,rtol=1e-8,history=true)
    return x,y,reshape(u,m+1,n+1),stats
end
```

```{code-cell}
f = x -> -sin(3x[1]*x[2]-4x[2]) * (9x[2]^2+(3x[1]-4)^2)
g = x -> sin(3x[1]*x[2]-4x[2])
xspan = [0,1];  yspan = [0,2];
x,y,U,stats = iterative(50,60,xspan,yspan,f,g)
println(stats)
```

```{code-cell}
using Plots
surface(x,y,U',color=:viridis,
    title="Solution of Poisson's equation",      
    xaxis=("x"),yaxis=("y"),zaxis=("u(x,y)"),camera=(120,50))    
```

```{code-cell}
Û = [g([x,y]) for x in x, y in y]
contour(x,y,(U-Û)',color=:bluesreds,aspect_ratio=1,
    title="Error",clims=(-0.0008,0.0008),      
    xaxis=("x"),yaxis=("y"),zaxis=("u(x,y)"),
    right_margin=10Plots.mm)   
```

```{code-cell}
gr()
m = 20:20:120
plt = plot()
for m in m
    x,y,U,stats = iterative(m,m,xspan,yspan,f,g);
    res = stats.residuals
    res = @. max(res,eps())
    plot!(0:length(res)-1,log10.(res/res[1]))
    # println(length(res))
end
plt
```

```{code-cell}
default(size=(400,200))
m = 20
x,y,U,stats = iterative(m,m,xspan,yspan,f,g);
res = stats.residuals
res = @. min(max(res,eps()),1)
plot(0:length(res)-1,res/res[1],yscale=)
    # println(length(res))
```

```{code-cell}
function iterative(m,n,xspan,yspan,f,g,prec)
    function precon(u)
        A = I + 0.1Dxx
        u = vec(A\reshape(u,m+1,n+1))      
        B = I + 0.1Dyy
        u = vec(reshape(u,m+1,n+1)/B)
        return u
    end

    x,Dx,Dxx = diffmats(m,xspan...)
    y,Dy,Dyy = diffmats(n,yspan...)
    grid = [(x,y) for x in x, y in y]

    # Identify boundary locations.
    isboundary = trues(m+1,n+1)
    isboundary[2:m,2:n] .= false
    idx = vec(isboundary);
    isinter = .!

    # forcing function / boundary values vector
    b = vec( f.(grid) )
    b[idx] = g.(grid[idx]);   # assigned values
    
    # Apply Laplacian operator with Dirichlet condition.
    function laplacian(v)
        V = reshape(v,m+1,n+1)
        AV = Dxx*V + V*Dyy'
        AV[idx] .= V[idx]   # Dirichlet condition
        return vec(AV)
    end

    A = LinearMap(laplacian,(m+1)*(n+1))
    M = LinearMap(precon,(m+1)*(n+1))
    u,stats = gmres(A,b,rtol=1e-8,history=true,M=M)
    return x,y,reshape(u,m+1,n+1),stats
end 

m = 20:20:100
plt = plot()
for m in m
    x,y,U,stats = iterative(m,m,xspan,yspan,f,g,true);
    res = stats.residuals
    plot!(0:length(res)-1,log10.(res/res[1]),label="m=$m",m=3,xlabel="iteration")
end
plt
```

```{code-cell}

```
