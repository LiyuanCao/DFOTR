function o = phi(x, type)
% basis function 
% The input is a column vector x and the return is a row vector phi(x). 

n = length(x);


if strcmp(type, 'linear')
    o = [1, x'];
elseif strcmp(type, 'quadratic')
    o = nan(1, (n+1)*(n+2)/2); 
    o(1) = 1;
    o(2:n+1) = x;
    temp = x * x' + (1/sqrt(2) - 1) * diag(x.^2);
    o(n+2:end) = temp(tril(true(n)));
end




end