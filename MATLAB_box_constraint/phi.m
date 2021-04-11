function o = phi(x, type)
% basis function 
% If the input is a column vector x, return row vector phi(x). 
% If the input is a matrix x, return a matrix such that the 
% ith row is phi(x(i,:)).

[r,c] = size(x); 

if c == 1
    n = r;

    if strcmp(type, 'linear')
        o = [1, x'];
    elseif strcmp(type, 'quadratic')
        o = nan(1, (n+1)*(n+2)/2); 
        o(1) = 1;
        o(2:n+1) = x;
        temp = x * x' + (1/sqrt(2) - 1) * diag(x.^2);
        o(n+2:end) = temp(tril(true(n)));
    end
    
elseif c > 1
    n = c;

    if strcmp(type, 'linear')
        o = [ones(r,1), x];
    elseif strcmp(type, 'quadratic')
        o = nan(r, (n+1)*(n+2)/2); 
        o(:,1) = 1;
        o(:,2:n+1) = x;
        for i = 1:r
            temp = x(i,:)' * x(i,:) + (1/sqrt(2) - 1) * diag(x(i,:).^2);
            o(i, n+2:end) = temp(tril(true(n)));
        end
    end
    
end


end