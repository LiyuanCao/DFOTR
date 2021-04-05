classdef dfo_tr < handle
% This is a derivative free trust region optimization algorithm. 
% This algorithm is designed to minimize blackbox functions.
% 
% Written by Liyuan Cao @Lehigh University in 2021.


properties
    n % [positive integer] input dimension
    options % [struct] options that control the behavior of the algorithm
    info % [struct] logger
    samp % [object] sample set 
    model % [object] approximation model 
end

methods
    function self = dfo_tr(x0, options)
        % input dimension
        self.n = length(x0);
        
        % default options
        % the algorithm
        self.options.alg_model = 'quadratic'; % the model type
        self.options.alg_TR = 'ball'; % trust region type
        self.options.alg_TRsub = 'exact'; % trust region subproblem
        % sample management parameters
        self.options.sample_initial = self.n + 1; % initial sample size
        self.options.sample_min = ceil((self.n+1) * 1.1); % minimum sample size
        self.options.sample_max = min(3000, (self.n+1)*(self.n+2)/2); % maximum sample size
        self.options.sample_toremove = 50; % a point is to be removed from the sample set, if it is sample_toremove * tr_delta away from the current point
        % trust region parameters
        self.options.tr_delta = 1.0; % initial delta (i.e. trust region radius)
        self.options.tr_toaccept = eps; % rho level to accept new point
        self.options.tr_toexpand = 0.5; % rho level to expand radius
        self.options.tr_expand = 1.3;   % radius expansion factor
        self.options.tr_toshrink = -5e-3;   % To avoid issues caused by extremely bad models, TR radius does not shrink when tr_toshrink <= rho < 0. 
        self.options.tr_shrink = 0.5^(10 / (10+self.n)); % radius shrink factor
        % stopping crieria parameters
        self.options.stop_iter = 2000; % maximum number of iterations
        self.options.stop_nfeval = 2000; % maximum number of function evaluations
        self.options.stop_delta = 1e-6; % throshold for trust region radius
        self.options.stop_predict = 1e-8; % threshold for predicted decrease
        % verbosity
        self.options.verbose = 2; 
        
        % Override default options with custom ones.
        if nargin > 1 && ~isempty(options)
            option_names = fieldnames(options);
            for i = 1:length(option_names)
                if isfield(self.options, option_names{i})
                    self.options.(option_names{i}) = options.(option_names{i}); 
                else
                    error([option_names{i}, ' is not a valid option name'])
                end
            end
        end
        
        % information/logger
        self.info.start_time = clock;
        self.info.iteration = 0;
        self.info.nsuccess = 0; 
        self.info.nfeval = 0; 

        % Create initial sample. 
        self.samp = Sample(x0, self.options);

        % Initialize model. 
        self.model = ApproximationModel(self.n, self.options);
    end

    function o = ask(self)
        % Return the first not evaluated point in the sample set.
        idx = find(isnan(self.samp.fY), 1);
        o = self.samp.Y(idx,:)'; 
    end

    function tell(self, x, fx)
        % Store the newly acquired function value. 
        idx = find(all(self.samp.Y==x', 2) & isnan(self.samp.fY), 1); 
        self.samp.fY(idx) = fx; 
        self.info.nfeval = self.info.nfeval + 1;

        % If there is still a point that is not evaluated, we go to evaluate it. 
        if any(isnan(self.samp.fY))
            return
        end

        % If all points are evaluated, we begin the optimization process.
        self.suggest();
    end
    
    function suggest(self)
        % Run one DFO-TR iteration.
        
        if self.info.iteration == 0
            % Put center of the initial trust region at the point in the
            % initial sample with the lowest function value. 
            [~,idx] = min(self.samp.fY); 
            self.model.center = self.samp.Y(idx,:)';
            self.model.delta = self.options.tr_delta;

            if self.options.verbose >= 2
                fprintf("\n Iteration Report \n")
                fprintf('|  iter |suc|  objective  | TR_radius |    rho    | m  |\n')
                fprintf("| %5d |---| %11.5e | %9.6f | --------- | %d |\n", ...
                    self.info.iteration, min(self.samp.fY),...
                    self.model.delta, self.samp.m)
            end

        else 
            % Calculate the ratio between the actual reduction in function 
            % value and the reduction predicted the approximation model. 
            rho = (self.model.c - self.samp.fY(end)) / self.info.predicted_decrease;

            % Calculate the ratio between the step size and the trust region radius.
            stepSize = norm(self.samp.Y(end,:)' - self.model.center);
            stepSize2delta = stepSize / self.model.delta;

            % Decide whether to move the iterate. 
            if rho >= self.options.tr_toaccept
                success = 1;
                self.info.nsuccess = self.info.nsuccess + 1;
                self.model.center = self.samp.Y(end,:)';
            else
                success = 0;
            end

            % Update the trust region radius. 
            self.model.update_delta(rho, stepSize2delta, self.options)

            % Remove points that are too far away from the current TR. 
            self.samp.auto_delete(self.model, self.options)

            % print iteration report
            if self.options.verbose >= 2
                fprintf("| %5d | %d | %11.5e | %9.6f | %9.6f | %d |\n", ...
                    self.info.iteration, success, self.samp.fY(end),...
                    self.model.delta, rho, self.samp.m)
            end
        end

        % build an approximation model
        self.model.fit(self.samp)

        % Solve the trust region subproblem. 
        [x1, self.info.predicted_decrease] = self.model.minimize();

        % Add the new point to the sample set. 
        self.samp.addpoint(x1) % the new point
        self.info.iteration = self.info.iteration + 1;
    end
    
    function STOP = stop(self)
        STOP = false;
        if self.info.iteration == 0
            return;
        end

        if self.info.iteration >= self.options.stop_iter
            STOP = true;
            self.info.exit_message = 'Exiting because the maximum number of iterations is reached.';
        elseif self.info.nfeval >= self.options.stop_nfeval
            STOP = true;
            self.info.exit_message = 'Exiting because the maximum number of function evaluations is reached.';
        elseif self.model.delta <= self.options.stop_delta
            STOP = true;
            self.info.exit_message = 'Exiting because the minimum trust region radius is reached.';
        elseif self.info.predicted_decrease <= self.options.stop_predict
            STOP = true;
            self.info.exit_message = 'Exiting because the minimum predicted decrease is reached.';
        end
        
        if ~STOP, return; end
        
        % log the best solution
        [self.info.bestfx, idx] = min(self.samp.fY);
        self.info.bestx = self.samp.Y(idx,:)';
        self.info = rmfield(self.info, 'predicted_decrease');
        
        if self.options.verbose >= 1
            fprintf('%s\n', self.info.exit_message)
            fprintf('***************** FINAL REPORT ************************\n')
            self.info.end_time = clock;
            fprintf('total elapsed time: %f seconds\n',...
                etime(self.info.end_time, self.info.start_time));
            fprintf("|#iter|#success|#fevals| best fvalue |final tr_radius|\n")
            fprintf("|%5d| %5d  | %5d | %11.5e |   %9.6f   |\n", ...
                self.info.iteration, self.info.nsuccess, self.info.nfeval,...
                 self.info.bestfx, self.model.delta);
        end
    end
end
    
methods (Static)
    function [bestx, bestfx, info] = optimize(func, x0, options)
        optimizer = dfo_tr(x0, options); 
        
        while true
           x = optimizer.ask();
           fx = func(x);
           optimizer.tell(x, fx);
           if optimizer.stop(), break; end
        end
        
        info = optimizer.info;
        bestfx = info.bestfx;
        bestx = info.bestx;
    end
end
end

