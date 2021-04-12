classdef Sample < handle
% sample set management
% 
% Written by Liyuan Cao @Lehigh University in 2021.

properties
    n % input dimension
    Y % sample set in the input space (Each row is a point in R^n.)
    fY % sample set function values
    options
end

properties (Dependent)
    m % sample size 
end

methods
    
    function self = Sample(x, options)
        % input dimension
        self.n = length(x);
        
        % options 
        self.options.sample_initial = options.sample_initial; % initial sample size
        self.options.sample_min = options.sample_min; % minimum sample size
        self.options.sample_max = options.sample_max; % maximum sample size
        self.options.sample_toremove = options.sample_toremove; % a point is to be removed from the sample set, if it is sample_toremove * tr_delta away from the current point
        if strcmp(options.alg_TR, 'ball')
            self.options.norm = 2;
        elseif strcmp(options.alg_TR, 'box')
            self.options.norm = inf;
        end
        
        % initial sample
        Y = x + options.tr_delta * eye(self.n);
        self.Y = [Y; x'];

        % function value of initial sample
        self.fY = nan(size(self.Y, 1), 1);
    end
    
    function o = get.m(self)
        o = size(self.Y, 1);
        if o ~= length(self.fY)
            error('Number of sample points and number of function values do not match!')
        end
    end

    function value = Ycentered(self, center, indices)
        % Y after the origin is shifted to a specified center 
        if nargin <= 2, indices = 1:self.m; end
        value = self.Y(indices,:) - center';
    end

    function value = distance(self, center, indices)
        % Euclidean distance from each sample point to a specified center 
        if nargin <= 2, indices = 1:self.m; end
        value = vecnorm(self.Ycentered(center, indices), self.options.norm, 2);
    end

    function idx = addpoint(self, x, model)
        % Add a point to the sample set. 
        % If the sample set has already reached its capacity, 
        % also return the index of the point that should be removed. 
        
        % Add the new point. 
        self.Y = [self.Y; x'];
        self.fY = [self.fY; nan];
        
        % sample set in the basis
        phix = phi(x - model.center, model.type.model);
        Phi = nan(self.m, length(phix));
        for i = 1:self.m
            Phi(i,:) = phi(self.Ycentered(model.center, i)', model.type.model);
        end
        
        if self.m > self.options.sample_max
            l = Phi(1:self.m-1,:)' \ phix';
            if strcmp(model.type.model, 'linear')
                l = abs(l) .* self.distance(model.center, 1:self.m-1).^2;
            elseif strcmp(model.type.model, 'quadratic')
                l = abs(l) .* self.distance(model.center, 1:self.m-1).^3;
            end
            
            [~,idx] = max(l);
            return
        end
        
        % If the sample set is still within the capacity, but the new point
        % will render it not poised, pick a point to remove so that the new
        % sample set is as well poised as possible in the subspace. 
        if rank(Phi) < self.m
            [Q,R] = qr(Phi(1:self.m-1,:)');
            Phi_subspace = R'; 
            phixnew = Q' * phix';
            l = Phi_subspace' \ phixnew;
            if strcmp(model.type.model, 'linear')
                l = abs(l) .* self.distance(model.center, self.m).^2;
            elseif strcmp(model.type.model, 'quadratic')
                l = abs(l) .* self.distance(model.center, self.m).^3;
            end
            
            [~,idx] = max(l);
            return
        end
        
        % If no point needs to be removed. 
        idx = [];
    end
    
    function rmpoint(self, idx)
        % remove points
        self.Y(idx,:) = [];
        self.fY(idx) = [];
    end
    
    function auto_delete(self, model, options)
        % Delete points that are too far away from the trust region.
        % After the deletion, the sample size should be no less than
        %  sample_min, and no more than sample_max.

        distance = self.distance(model.center);

        % indeces of the ones that are too far
        toofar = find(distance > options.sample_toremove * model.delta);
        % indeces of the farthest ones
        % We should not delete too many points so that sample size goes under sample_min. 
        [~,farthest] = maxk(distance, max(self.m - options.sample_min,0));
        % indeces of the points to be removed
        if isempty(toofar) && self.m > options.sample_max
            % We need to keep the sample size under its maximum. 
            toremove = farthest(1:self.m-options.sample_max);
        else
            toremove = intersect(farthest, toofar);
        end

        % remove points
        self.rmpoint(toremove);
    end
end
end


function output = quadratic_terms(x)
    n = length(x);
    temp = x * x' - (1 - 1/sqrt(2)) * diag(x.^2);
    output = temp(tril(ones(n)));
end