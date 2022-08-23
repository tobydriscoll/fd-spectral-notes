using LinearSolve
include("diffmats.jl")
function poissoncg(f,g,m,xspan,n,yspan)
    # Discretize the domain.
    x,Dx,Dxx = diffmats(m,xspan...)
    y,Dy,Dyy = diffmats(n,yspan...)
    grid = [(x,y) for x in x, y in y]

    # Identify boundary locations.
    isboundary = trues(m+1,n+1)
    isboundary[2:m,2:n] .= false
    idx = vec(isboundary)

    # Apply Laplacian operator with Dirichlet condition.
    function laplacian(v)
        V = reshape(v,m+1,n+1)
        AV = Dxx*V + V*Dyy'
        AV[idx] .= V[idx]   # Dirichlet condition
        return vec(AV)
    end
    
    # Solve the linear system and reshape the output.
    b = vec( f.(grid) )
    b[idx] = g.(grid[idx])    # assigned values
    system = LinearProblem((u,p,t)->laplacian(v),b)
    u = solve(system).u
    U = reshape(u,m+1,n+1)
    return x,y,U
end