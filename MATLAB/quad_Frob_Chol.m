function [H,g,L] = quad_Frob_Chol(Y,f_values,L0)

[n,m] = size(Y);
L = [];

if m <= (1+n)*n/2
    if isempty(L0)

        % cleverly calculate A = MQ * MQ'
        A = ((Y'*Y).^2)/2;

        % Cholesky decomposition
        % if A is not pd due to numerical issue, add diagonal matrix
        % until it is pd
        A = A + 1e-11*eye(m); 
        while 1
            try
                L = chol(A)';
                break;
            catch
                A = A + 1e-10*eye(m); 
            end
        end

    else
        % calculate the new row/column for matrix A = MQ * MQ'
        Anew = (( Y'*Y(:,end) ).^2)/2;

        % calculate the new row/column for matrix L  (A = L * L')
        Lnew = NaN(m,1);

        Lnew(1:m-1) = L0 \ Anew(1:m-1);

        if Anew(m) - Lnew(1:m-1)' * Lnew(1:m-1) < eps
%             error('Anew(m) - Lnew(1:m-1)'' * Lnew(1:m-1) = %f < 0',...
%                 Anew(m) - Lnew(1:m-1)' * Lnew(1:m-1))
            [H,g,L] = quad_Frob_Chol(Y,f_values,[]);
            return;
        end
        Lnew(m) = sqrt( Anew(m) - Lnew(1:m-1)' * Lnew(1:m-1) );

        % the new L
        L = [L0 zeros(m-1,1); Lnew(1:m)'];

    end
    NQ = L' \ (L\[Y' f_values']);
    bQ = NQ(:,end);
    NQ = NQ(:,1:n);
    NL = Y * NQ;
    % bQ = L' \ (L\F_values');
    bL = Y * bQ;

    %  Retrieve the model coefficients.
    g = NL \ bL; 
    H = Y * bsxfun(@times, bQ - NQ*g, Y');
    
    if ~isreal(H) || ~isreal(g)
        temp = 0;
    end
    
% elseif m < (n+1)*(n+2)/2
%     
%         A = ((Y'*Y).^2)/2;
%         W = [A Y' ones(m,1); [Y; ones(1,m)] zeros(n+1)];
%        
%        %  Compute the model coefficients.
% %    warning('off', 'MATLAB:nearlySingularMatrix')
%         lambda         = W \ [F_values 0*rand(1,n+1)]';
%    
% %  Retrieve the model coefficients.
%         g = lambda(m+1:m+n);
%         H = Y * bsxfun(@times, lambda(1:m), Y');
% 
% else
% 	phi_Q = NaN(m, n*(n+1)/2);
% 	for i = 1:m
%         y      = Y(:,i);
%         aux_H  = y*y'-1/2*diag(y.^2);
%         aux    = aux_H( tril(true(n)) ); % vectorized lower triangular part
%         phi_Q(i,:) = aux';
% 	end
%     W = [ones(m,1) Y' phi_Q];
%     
%     %  Compute the coefficients of the model.
% %    warning('off', 'MATLAB:nearlySingularMatrix')
% %    if size(W,1)~=size(W,2), warning('W is not a square matrix.'); end
% 	lambda         = W \ F_values';
% 
% %  Retrieve the model coefficients.
%     g    = lambda(2:n+1);
%     H    = zeros(n);
%     H( tril(true(n)) ) = lambda(n+2:end);
%     H = H + H' - diag(diag(H));

else
    error('This part is not finished yet.')
end



% End of quad_Frob.