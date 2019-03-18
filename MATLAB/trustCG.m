function [s,val] = trustCG(g,H,delta)

% trust region subproblem solver, uses conjugate gradient method (CG)
% input: g        = [n-by-1] gradient at trust region center
%        H        = [n-by-n] hessian at trust region center
%        delta    = [scalar] trust region radius
% outpu: s        = [n-by-1] step
%        val      = [scalar] new function value
%
% Written by Liyuan Cao @Lehigh University in October 2017.

n = length(g);

s = zeros(n,1);
r = H*s + g; % residual
p = -r; 

for k = 1:n
    if p'*H*p <= 0
        a = p'*p;
        b = s'*p * 2;
        c = s'*s - delta^2;
        alpha = (-b + sqrt(b^2 - 4*a*c)) / 2 / a;
        break;
    else 
        alpha = r'*r / (p'*H*p);
    end
    
    if norm(s + alpha*p) > delta
        a = p'*p;
        b = s'*p * 2;
        c = s'*s - delta^2;
        alpha = (-b + sqrt(b^2 - 4*a*c)) / 2 / a;
        break;
    else 
        s = s + alpha*p;
        r0 = r;
        r = r + alpha*(H*p);
    end
    
    if norm(r) < 1e-5
        break; 
    else
        p = -r + (r'*r) / (r0'*r0) * p;
    end
end

s = s + alpha*p;
val = g'*s + s'*H*s;