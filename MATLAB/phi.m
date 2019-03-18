function out = phi(x, type)
% returns the monomial basis of x. 
% type defines whether the basis is quadratic or linear. 

n = length(x);
x = x(:);

if strcmp(type, 'quadratic')

    out = nan(1 + n + (1+n)*n/2, 1);

    out(1) = 1;

    out(2:n+1) = x;

    temp = x*x';
    out(n+2:end) = temp(triu(true(n)));
    
elseif strcmp(type, 'linear')
    
    out = [1; x];
end