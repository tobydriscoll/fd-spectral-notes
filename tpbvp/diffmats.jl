using LinearAlgebra
function diffmats(x)
    # assumes evenly spaced nodes
    h = x[2]-x[1]
    m = length(x)
    Dx = 1/2h*diagm(-1=>[-ones(m-2);-2],0=>[-2;zeros(m-2);2],1=>[2;ones(m-2)])
    Dxx = 1/h^2*diagm(-1=>[ones(m-2);-2],0=>[1;-2*ones(m-2);1],1=>[-2;ones(m-2)])
    Dxx[m,m-2] = Dxx[1,3] = 1/h^2
    return Dx,Dxx
end
