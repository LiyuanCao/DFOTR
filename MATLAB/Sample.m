classdef Sample < handle
% sample set management
% 
% Written by Liyuan Cao @Lehigh University in 2021.

properties
    n % input dimension
    Y % sample set in the input space (Each row is a point in R^n.)
    fY % sample set function values
end

properties (Dependent)
    m % sample size 
end

methods
    
    function self = Sample(x, options)
        % input dimension
        self.n = length(x);

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

    function value = Ycentered(self, center)
        % Y after the origin is shifted to a specified center 
        value = self.Y - center';
    end

    function value = distance(self, center)
        % Euclidean distance from each sample point to a specified center 
        value = vecnorm(self.Ycentered(center), 2, 2);
    end

    function addpoint(self, point)
%         if self.m >= options.sample_max
%             error('Attemp to add more points when the sample set is already at its capacity. ')
%         end
        self.Y = [self.Y; point'];
        self.fY = [self.fY; nan];
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
        self.Y(toremove,:) = [];
        self.fY(toremove) = [];
    end
end
end


function output = quadratic_terms(x)
    n = length(x);
    temp = x * x' - (1 - 1/sqrt(2)) * diag(x.^2);
    output = temp(tril(ones(n)));
end