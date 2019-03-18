classdef interpolationModel
    % class: interpolation model
    
    properties
        type
        coefficients
    end
    
    methods 
        function obj = interpolationModel(samp, par)
            % initialization: build the interpolation model 
            
            % basic numbers
            m = samp.m - 1;
            n = samp.n;

            % evaluate the model coefficients
            if strcmp(par.algorithm.model, 'quadratic') && (m > n)
                % fit a quadratic model 
                obj.type = 'quadratic';

                % construct the big matrix
                A = nan(n+m, n+m);
                A(1:n, 1:n) = zeros(n);
                A(1:n, n+1:end) = samp.Ycentered';
                A(n+1:end, 1:n) = samp.Ycentered; 
                A(n+1:end, n+1:end) = (samp.Ycentered * samp.Ycentered').^2 / 2;

                % the right-hand side
                b = zeros(n+m, 1);
                b(n+1:end) = samp.fcentered;

                % solve the linear system and retrieve the gradient and the hessian 
                lambda = A \ b;
                if any(isnan(lambda))
                    % in case matrx A is singular
                    [U,S,V] = svd(A);
                    
                    s = diag(S);
                    s(abs(s) > 1e-5) = 1 ./ s(abs(s) > 1e-5);
                    s(abs(s) <= 1e-5) = 0;
                    lambda = V*diag(s)*U' * b;
                end
                obj.coefficients.g = lambda(1:n);
                obj.coefficients.H = samp.Ycentered' * ...
                    bsxfun(@times, lambda(n+1:end), samp.Ycentered);
                

            elseif strcmp(par.algorithm.model, 'linear') || (m <= n)
                % fit a well-determined or an over-determined linear model 
                obj.type = 'linear';

                % linear regression
                obj.coefficients.g = samp.Ycentered \ samp.fcentered;
                
                obj.coefficients.H = zeros(n);
                
            else
                error('Something is wrong.')
            end
            
            % a safety net
            if any(isnan(obj.coefficients.g)) || any(isnan(obj.coefficients.H(:)))
                obj.coefficients.g = rand(n,1) - 0.5;
                obj.coefficients.H = zeros(n);
                warning('There is at one ''nan'' in the coeffcients of the model. \n')
            end
            
        end
        
        function [s, val] = minimum_point(obj, delta, par)
            % returns the minimum point of this model within a trust region
            % requires the trust region radius delta
            
            if strcmp(obj.type, 'linear') ||...
                    norm(obj.coefficients.H,'fro') / norm(obj.coefficients.g) < 1e-3
                s = - delta / norm(obj.coefficients.g) * obj.coefficients.g;
                val = obj.coefficients.g' * s;
                return;
            end
            
            if strcmp(par.algorithm.tr, 'exact')
                [s,val] = trust(obj.coefficients.g, obj.coefficients.H, delta);
                if any(~isreal(s)) || norm(s) > 2*delta
                    % Just in case trust fails, we go back to linear model.
                    s = - delta / norm(obj.coefficients.g) * obj.coefficients.g;
                	val = obj.coefficients.g' * s;
                end
            elseif strcmp(par.algorithm.tr, 'CG')
                [s,val] = trustCG(obj.coefficients.g, obj.coefficients.H, delta);
            else
                error('unidentified trust region algorithm \n')
            end
        end
    end
end










