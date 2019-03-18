classdef sample
    properties
        Y
        f
        centeridx
        Ycentered
        fcentered
        n
        m
        mgood
        distance
        % * Ycentered, fcentered, and distance do not include the center. 
    end
    
    methods
        
       function obj = sample(x, par)
           % initialization
           
            % input dimension
            obj.n = par.n;

            % initial sample
            Y = bsxfun(@plus, x', par.tr.delta * [eye(obj.n); -eye(obj.n)]);
%             Y = mvnrnd(x, par.tr.delta / 3 * eye(obj.n), par.sample.initial - 1);
            obj.Y = [x'; Y]; 

            % initial sample size
            obj.m = size(obj.Y,1);

            % number of sample points within the acceptable range
            obj.mgood = obj.m;

            %function value of initial sample
            obj.f = nan(obj.m, 1);

       end
       
       function obj = centering(obj, centeridx)
           % centering
           % This is also the second stage of the initialization.
           
            % centering origin
            obj.centeridx = centeridx;

            % centered sample
            obj.Ycentered = bsxfun(@minus, ...
                obj.Y([1:obj.centeridx-1, obj.centeridx+1:end], :), ...
                obj.Y(centeridx, :) );
            obj.fcentered = obj.f([1:obj.centeridx-1, obj.centeridx+1:end]) - obj.f(centeridx);

            % distance of each point in the sample (excluding the center)
            % to the center
            obj.distance = vecnorm(obj.Ycentered, 2, 2);
       end
       
       function [obj, indeces] = addpoints(obj, points, fvalues, delta, par)
           % add new points 
           % After adding the new point, the sample size may exceed
           % par.sample.max. 
           % Returns the new sample set and the indeces of the newly added
           % points in the sample set. 
           
            % number of new points
            p = size(points, 1);

            % sample size 
            obj.m = obj.m + p;

            % sample
            obj.Y = [obj.Y; points];

            % centered points 
            centeredPoints = bsxfun(@minus, points, obj.Y(obj.centeridx,:));
            obj.Ycentered = [obj.Ycentered; centeredPoints];

            % function values
            obj.f = [obj.f; fvalues];

            % centered function values
            obj.fcentered = [obj.fcentered; fvalues - obj.f(obj.centeridx)];

            % distance to the center
            d = vecnorm(centeredPoints, 2, 2);
            obj.distance = [obj.distance; d];

            % number of sample points within the acceptable range
            obj.mgood = obj.mgood + sum(d < delta * par.sample.toremove);
            
            % indeces of the new points
            indeces = (obj.m - p + 1 : obj.m)';
        end

       
       function [obj, indeces] = deletepoints(obj, delta, par)
            % remove points that are too far away from the center  
            % After the deletion, the sample size should be no less than
            % par.sample.min, and no more than par.sample_max.
            % Returns the new sample set and a list for referencing the new indeces. 
            % newindex = indeces(oldindex) 
            % This newindex is 0 is the the point corresponding to the
            % oldindex is deleted. 
            
            % indeces of the farthest ones; 
            % We should not delete too many points so that sample size goes under par.sample_min.
            [~, farthest] = maxk(obj.distance, obj.m - par.sample.min);
            
            % indeces of the ones that are too far
            toofar = find(obj.distance > par.sample.toremove * delta);
            
            % indeces of the points to be removed
            if isempty(toofar) && obj.m > par.sample.max
                % We need to keep the sample size under its maximum. 
                toremove = farthest(1:(obj.m - par.sample.max));
            else
                toremove = intersect(farthest, toofar);
            end
            
            % remove points
            obj.Ycentered(toremove, :) = [];
            obj.fcentered(toremove, :) = [];
            obj.distance(toremove, :) = [];
            toremove1 = toremove;
            toremove1(toremove >= obj.centeridx) = toremove1(toremove >= obj.centeridx) + 1;
            obj.Y(toremove1,:) = [];
            obj.f(toremove1,:) = [];

            % update sample size
            obj.m = size(obj.Y, 1);
            obj.mgood = obj.m + length(toremove) - length(toofar);
            
            % the indeces referencing list
            indeces = nan(obj.m + length(toremove), 1);
            removed = false(obj.m + length(toremove), 1);
            removed(toremove1) = true;
            indeces(removed) = 0;
            indeces(~removed) = 1:obj.m;
            
            % update the center index
            obj.centeridx = indeces(obj.centeridx);
       end

       function points = poisedness_improvement_points(obj, p, delta, par)
            % poisedness improvement points
            % p is the number of points requested 
            
            Y = [zeros(1, obj.n); obj.Ycentered];
            
            delta = 0.75 * delta * par.sample.toremove;
            
            if strcmp(par.algorithm.model, 'linear')
                % the features
                Phi = nan(obj.m, 1 + obj.n);
                % the Lagrangian polynomials
                A = eye(1 + obj.n, obj.m + p);
            elseif strcmp(par.algorithm.model, 'quadratic')
                % the features
                Phi = nan(obj.m, 1 + obj.n + (1+obj.n)*obj.n/2);
                % the Lagrangian polynomials
                A = eye(1 + obj.n + (1+obj.n)*obj.n/2, obj.m + p);
            end
            
            for i = 1:obj.m
                Phi(i,:) = phi(Y(i,:)', par.algorithm.model);
            end
            points = [];
            
            
            
            for i = 1:size(A, 2)
                lis = Phi * A(:,i);
    
                [v,ji] = max(abs(lis(i:end)));
                ji = ji + i - 1;
                if isempty(v), v = 0; end

                if v > 0.001 && i <= obj.m
                    Y([i, ji], :) = Y([ji, i], :);
                    Phi([i, ji], :) = Phi([ji, i], :);
 
                else
                    if i <= obj.m
                        a = 1;
                    end
                    if strcmp(par.algorithm.model, 'linear')
                        g = A(2:obj.n+1,i);
                        
                        y1 = - delta / norm(g) * g;
                        val1 = A(1,i) + g'*y1;
                        y2 = delta / norm(g) * g;
                        val2 = A(1,i) + g'*y2;
                        
                    elseif strcmp(par.algorithm.model, 'quadratic')
                        g = A(2:obj.n+1,i);
                        H = zeros(obj.n);
                        H(triu(true(obj.n))) = A(obj.n+2:end,i);
                        H = H + H';
                        [y1, val1] = trust(-g, -H, delta);
                        val1 = A(1,i) - val1;
                        [y2, val2] = trust(g, H, delta);
                        val2 = A(1,i) + val2;
                    end

                    if abs(val1) > abs(val2)
                        Y(i, :) = y1;
                        Phi(i, :) = phi(y1, par.algorithm.model);
                        points = [points; y1'];
                    else
                        Y(i, :) = y2;
                        Phi(i, :) = phi(y2, par.algorithm.model);
                        points = [points; y2'];
                    end
%                     break;
                end

                % normalization
                A(:,i) = A(:,i) / ( Phi(i,:) * A(:,i) );

                % orthogonalization
                A(:, i+1:size(A,2)) = A(:, i+1:size(A,2)) ...
                    - A(:,i) * (Phi(i,:) * A(:, i+1:size(A,2)));
            end
       end
    end
end












