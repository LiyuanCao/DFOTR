function [s, val] = trust_region_subproblem_solver(H,g,delta)

% Author      : Liyuan 'Leon' Cao
% Description : Solves trust region subproblem, calculates lambda and d at
%               the same time
% Input       : g = [n-by-1 vector] the gradient of current iterate
%               H = [n-by-n matrix] the Hessian matrix
%               delta = [scalar] trust region radius
% Output      : s = [vector] step 

lambda = eig(H);
lambda = -min(lambda);

while 1
        
    % Cholesky factorization, H+lambda*I = U'U, where U is a upper
    % triangular matrix
    B = H + (lambda + 1e-10) * eye(length(H));
    while true
        try
            U = chol(B);
            break;
        catch
            B = B + (lambda + 1e-10) * eye(length(H));
        end
    end

    % Solve for d
    s = U' \ (U\(-g));

    % Solve for q
    q = zeros(length(s),1);
    for j = 1:length(s)
        q(j) = ( s(j)-U(1:j-1,j)'*q(1:j-1) ) / U(j,j);
    end

    % Update lambda
    lambda = lambda + (norm(s)/norm(q))^2*(norm(s)-delta)/delta;

    % Check for termination
    if norm(s)< delta + 1e-6, break; end
    
    if (norm(s)/norm(q))^2*(norm(s)-delta)/delta < 0
    end
end

val = g'*s + s'*H*s/2;
end