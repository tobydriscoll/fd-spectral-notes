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

# Discretizing the disk

```{code-cell}
û(x,y) = x^3 - 2y^2 + x*y

nr,nθ = 71,80
r = [ -1+i*2/nr for i in 0:nr ]
θ = [ i*π/nθ for i in 0:nθ-1 ]

X = [r*cos(θ) for r in r, θ in θ]
Y = [r*sin(θ) for r in r, θ in θ]
U = @. û(X,Y) 
```

```{code-cell}
pyplot()
default(size=(450,200))
surface(X,Y,U,color=:viridis,l=false)
```

```{code-cell}
[U reverse(U,dims=1)]
```

```{code-cell}
include("/Users/driscoll/817/notes/elliptic/diffmats.jl")
function polarlap(nr,nθ)
    @assert isodd(nr)
    ⊗ = kron
    r,Dr,Drr = diffmats(nr,-1,1)
    S = spdiagm(1 ./r)
    q = π/nθ
    θ = q*(0:nθ-1)
    Dθθ = 1/q^2*spdiagm(
            0=>fill(-2.,2nθ),
            -1=>ones(2nθ),
            1=>ones(2nθ),
            2nθ-1=>[1.],
            1-2nθ=>[1.]
        )
    Q₁₁,Q₁₂ = Dθθ[1:nθ,1:nθ],Dθθ[1:nθ,nθ+1:2nθ]
    L = I(nθ)⊗(Drr + S*Dr) + Q₁₁⊗S.^2 + Q₁₂⊗reverse(S.^2,dims=1)
    return r,θ,L
end

using Plots
r,θ,L = polarlap(11,20)
spy(L)
```

```{code-cell}
û(x,y) = x.^2 + 2y
f(x,y) = 2.0

nr,nθ = 39,58
r,θ,A = polarlap(nr,nθ)

X = [r*cos(θ) for r in r, θ in θ]
Y = [r*sin(θ) for r in r, θ in θ]

bdy = falses(nr+1,nθ)
bdy[[1,end],:] .= true 
bdy = vec(bdy)

N = (nr+1)*nθ
A[bdy,:] = I(N)[bdy,:]
b = f.(vec(X),vec(Y))
b[bdy] .= û.(vec(X)[bdy],vec(Y)[bdy])
u = A\b
U = reshape(u,nr+1,nθ);
```

```{code-cell}
# # alternate method--removal of boundary values
# u = zeros(size(A,1))
# u[bdy] = û.(X[bdy],Y[bdy])
# inter = @. !bdy
# Ã = A[inter,inter]
# f̃ = f.(X[inter],Y[inter]) - A[inter,bdy]*u[bdy]
# u[inter] = Ã\f̃;
```

```{code-cell}
pyplot()

surface(X,Y,U,color=:viridis,l=false)
```

```{code-cell}

```
