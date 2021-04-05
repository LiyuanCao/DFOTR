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
    center
    delta
end

methods
    function self = ApproximationModel(n, options)
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

        elseif strcmp(self.type.model, 'quadratic') && (m == n + 1)
            % fit an well-determined linear model 
            self.H(:) = 0;

            % solve the linear system
            A = nan(1+n); 
            A(:,1) = 1; 
            A(:,2:end) = Ycentered; 
            temp = A \ samp.fY; 
            self.c = temp(1); 
            self.g = temp(2:end); 
        end
    end

    function [minimizer, decrease] = minimize(self)
        if ~isreal(self.g)
            error('The gradient is not all real. ')
        end

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
        
        minimizer = self.center + s;
        decrease = -val;
    end

    function update_delta(self, rho, stepSize2delta, options)
        % Updating iterate and trust-region radius.

        if rho >= options.tr_toexpand && stepSize2delta >= 0.99
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




