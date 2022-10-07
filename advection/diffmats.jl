using LinearAlgebra,SparseArrays
function diffmats(n,a,b;periodic=false)
    # assumes evenly spaced nodes
    h = (b-a)/n
    if !periodic
        x = [ a+i*h for i in 0:n ]
        Dx = 1/2h*spdiagm(-1=>[-ones(n-1);-2],0=>[-2;zeros(n-1);2],1=>[2;ones(n-1)])
        Dxx = 1/h^2*spdiagm(-1=>[ones(n-1);-2],0=>[1;-2*ones(n-1);1],1=>[-2;ones(n-1)])
        Dxx[n+1,n-1] = Dxx[1,3] = 1/h^2
    else
        x = [ a+i*h for i in 0:n-1 ]
        Dx = 1/2h*spdiagm(1-n=>[1.],-1=>-ones(n-1),1=>ones(n-1),n-1=>[-1.])
        Dxx = 1/h^2*spdiagm(1-n=>[1.],-1=>ones(n-1),0=>fill(-2.,n),1=>ones(n-1),n-1=>[1.])
    end
    return x,Dx,Dxx
end
