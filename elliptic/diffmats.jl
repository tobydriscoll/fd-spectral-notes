using LinearAlgebra,SparseArrays
function diffmats(m,a,b)
    # assumes evenly spaced nodes
    h = (b-a)/m
    x = [ a+i*h for i in 0:m ]
    Dx = 1/2h*spdiagm(-1=>[-ones(m-1);-2],0=>[-2;zeros(m-1);2],1=>[2;ones(m-1)])
    Dxx = 1/h^2*spdiagm(-1=>[ones(m-1);-2],0=>[1;-2*ones(m-1);1],1=>[-2;ones(m-1)])
    Dxx[m+1,m-1] = Dxx[1,3] = 1/h^2
    return x,Dx,Dxx
end
