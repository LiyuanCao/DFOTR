classdef ApproximationModel < handle
% Contains a functin that build quadratic model based on the sample.
% 
% Written by Liyuan Cao @Lehigh University in 2021.
    
properties
    n % [positive integer] input dimension
    type % [struct] model type, trust region type, and TR subprobelm algorith
    c
    g
    H
    lb
    ub
    center % center of the trust region
    delta % radius of the trust region
end

methods
    function self = ApproximationModel(n, lb, ub, options)
        self.n = n;

        self.type.model = options.alg_model;
        self.type.TR = options.alg_TR;
        self.type.TRsub = options.alg_TRsub;

        % preallocate memory for approximation model
        if strcmp(self.type.model, 'linear')
            self.c = nan(1);
            self.g = nan(n,1);
        elseif strcmp(self.type.model, 'quadratic')
            self.c = nan(1);
            self.g = nan(n,1);
            self.H = nan(n);
        end
        
        % upper and lower bound for variables
        self.lb = lb;
        self.ub = ub;

        % preallocate memory for the trust region
        self.center = nan(n,1);
        self.delta = nan(1);
    end
        
    function fit(self, samp)
        if samp.n ~= self.n
            error('Dimensions of model and sample mismatch.')
        end
        m = samp.m;
        n = samp.n; 
        Ycentered = samp.Ycentered(self.center);

        if strcmp(self.type.model, 'quadratic') && (m > n + 1)
            % construct the big matrix
            A = nan(1+n+m, 1+n+m); 
            A(1:1+n, 1:1+n) = 0;
            A(1, 1+n+1:end) = 1;
            A(1+n+1:end, 1) = 1;
            A(2:1+n, 1+n+1:end) = Ycentered';  
            A(1+n+1:end, 2:1+n) = Ycentered; 
            A(1+n+1:end, 1+n+1:end) = (Ycentered * Ycentered').^2 / 2; 

            % the right-hand side
            b = zeros(1+n+m, 1);
            b(1+n+1:end) = samp.fY;

            % solve the linear system and retrieve the gradient and the hessian 
            lamda = A \ b;
            self.c = lamda(1); 
            self.g = lamda(2:1+n);
            self.H = Ycentered' * (lamda(1+n+1:end) .* Ycentered);
            self.H = (self.H + self.H') / 2;
            
            % The matrix A tends to be extremely poorly conditioned.
            % Just in case something goes wrong, we can fall back to linear
            % regression. 
            if isreal(lamda) && all(~isnan(lamda))
                return
            end
        end
        
        % fit an (over/well/under)-determined linear model 
        self.H(:) = 0;
        A = nan(m, 1+n); 
        A(:,1) = 1; 
        A(:,2:end) = Ycentered; 
        temp = A \ samp.fY; 
        self.c = temp(1); 
        self.g = temp(2:end); 
    end

    function [minimizer, decrease] = minimize(self)
        % Minimize the approximation model. 
        % Return the minimizer and its function value. 
        if ~isreal(self.g)
            error('The gradient is not all real. ')
        end

        if strcmp(self.type.TR, 'ball')
            if strcmp(self.type.model, 'linear') || (strcmp(self.type.model, 'quadratic')...
                 && (any(isnan(self.H(:))) || norm(self.H) < 1e-12))
                s = - self.delta / norm(self.g) * self.g; 
                val = self.g' * s; 

            elseif strcmp(self.type.model, 'quadratic') && strcmp(self.type.TRsub, 'exact')
                [s,val] = trust(self.g, self.H, self.delta);

            elseif strcmp(self.type.model, 'quadratic') && strcmp(self.type.TRsub, 'CG')
                [s,val] = trustCG(self.g, self.H, self.delta);

            else
                error('Something went wrong!')
            end
            
        elseif strcmp(self.type.TR, 'box')
            lb = max(self.lb - self.center, -self.delta * ones(self.n, 1));
            ub = min(self.ub - self.center, self.delta * ones(self.n, 1));
            if strcmp(self.type.model, 'linear') || (strcmp(self.type.model, 'quadratic')...
                 && (any(isnan(self.H(:))) || norm(self.H) < 1e-12))
            	options = optimoptions(@linprog, ...
                    'Display', 'off');
                if norm(self.g) > 0
                    [s,val] = linprog(self.g / norm(self.g), [], [], [], [], lb, ub, options); 
                    val = val * norm(self.g);
                else
                    s = zeros(self.n,1);
                    val = 0;
                end
                
            elseif strcmp(self.type.model, 'quadratic')
                options = optimoptions(@quadprog,...
                    'Algorithm','trust-region-reflective', ...
                    'Display', 'off');
                [s,val] = quadprog(self.H, self.g, [],[],[],[],lb,ub,zeros(self.n,1),options); 
                
            else
                error('Something went wrong!')
            end
            
        end
        
        minimizer = self.center + s;
        decrease = -val;
    end

    function update_delta(self, rho, x1, options)
        % Updating iterate and trust-region radius.
        
        % Check if the new iterate is on the boundary of the trust region. 
        if strcmp(self.type.TR, 'ball')
            on_boundary = norm(x1 - self.center) >= 0.99 * self.delta;
        elseif strcmp(self.type.TR, 'box')
            on_boundary = false; 
            if (norm(x1 - self.center, inf) >= 0.99 * self.delta) ||...
                    any(x1-self.lb < 1e-10) || any(self.ub-x1 < 1e-10)
                on_boundary = true;
            end
        end

        if rho >= options.tr_toexpand && on_boundary
            % When the approximation is very good, increase TR radius
            self.delta = self.delta * options.tr_expand;
        elseif (0 <= rho && rho < options.tr_toaccept) || (rho < options.tr_toshrink)
            % When the approximation is bad, but not complete bullshit, 
            % reduce the TR radius. 
            self.delta = self.delta * options.tr_shrink;
        end
    end
end
end




