include("diffmats.jl")
function poissonfd(f,g,m,xspan,n,yspan)
    # Discretize the domain.
    x,Dx,Dxx = diffmats(m,xspan...)
    y,Dy,Dyy = diffmats(n,yspan...)
    grid = [(x,y) for x in x, y in y]
    N = (m+1)*(n+1)   # total number of unknowns

    # Form the collocated PDE as a linear system.
    A = kron(I(n+1),Dxx) + kron(Dyy,I(m+1))
    b = vec( f.(grid) )

    # Identify boundary locations.
    isboundary = trues(m+1,n+1)
    isboundary[2:m,2:n] .= false
    idx = vec(isboundary)

    # Apply Dirichlet condition.
    scale = maximum(abs,A[n+2,:])
    A[idx,:] = scale * I(N)[idx,:]        # Dirichlet assignment
    b[idx] = scale * g.(grid[idx])    # assigned values

    # Solve the linear system and reshape the output.
    u = A\b
    U = reshape(u,m+1,n+1)
    return x,y,U
end
