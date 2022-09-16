using LinearAlgebra
function diffmats(a,b,n)
    # assumes evenly spaced nodes
    h = (b-a)/n
    x = [a + i*h for i in 0:n]
    Dx = 1/2h*diagm(-1=>[fill(-1.,n-1);-2],0=>[-2;zeros(n-1);2],1=>[2;ones(n-1)])
    Dxx = 1/h^2*diagm(-1=>[ones(n-1);-2],0=>[1;fill(-2.,n-1);1],1=>[-2;ones(n-1)])
    Dxx[n+1,n-1] = Dxx[1,2] = 1/h^2
    return x,Dx,Dxx
end
