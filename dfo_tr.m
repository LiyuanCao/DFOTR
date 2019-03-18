function [x,f] = dfo_tr(func, x, par)
%
% Function dfo_tr applies a derivative-free interpolation-based
% trust-region method to the problem:
%
%                   min f(x)
%
% where x is a real vector of dimension n.
%
% The user must provide func to evaluate the function f.
%
% Input:
%    fun      = [function] the objective function
%    x         = [n-by-1] the initial point to start the optimizer
%    param	= [optional, structure] for changing predefined
%                parameters, see initialize_paramters.m
%
% Output:
%    x         = [n-by-1] the optimal solution given by the solver
%    f         = [scalar] fun(x)
%
% Functions called: func (application, user provided),
%                   quad_Frob (provided by the optimizer),
%                   trust (provided by MATLAB).
% Written by K. Scheinberg and L. N. Vicente, 2010.
% Modified by Ruobing Chen, 2014 and Katya Scheinberg 2015.
% Modified by Liyuan Cao and Katya Scheinberg 2018.

fprintf('\n');
time = clock;

%%   initialization   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input dimension
n = length(x);

% initialize constant parameters
if nargin < 3
    par = [];
end
par = initialize_parameters(par, n);

% initial sample
samp = sample(x, par);
% evaluate f on the initial sample set - This may take a while.
samp.f = arrayfun(@(i) feval(func, samp.Y(i,:)'), (1:samp.m)');
% starting point index
[~,idx] = min(samp.f);
% centering 
samp = samp.centering(idx);


% initial trust region radius 
delta = par.tr.delta;

% Set counters.
iter      = 0; % iteration counter
iter_suc  = 0; % successful iteration counter
func_eval = samp.m; % function evaluation counter

% Print the iteration report header.
if par.print >= 2
    fprintf('Iteration Report: \n\n');
    fprintf('| iter  | success |     f_value     |');
    fprintf(' TR radius |    rho    |    m    |\n');
    print_format = '| %5d |    %2s   | %+13.8e | %+5.2e |\n';
    fprintf(print_format, iter, '--', samp.f(idx), delta);
end

%%   Start the iteration.   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while iter < par.stop.iter
    
    iter = iter + 1;
    
    % Build a quadratic model.
    model = interpolationModel(samp, par);
    
    % Run the trust region method on the interpolation model.
    [s, val] = model.minimum_point(delta, par);
    
    % Check a stopping criterion.
    if -val <= par.stop.tol_f
        % Stop if potential decrease is too small.
        fprintf('Existing due to small potential decrease. \n')
        break; 
    end

    % check out the new point
    m1 = samp.f(idx) + val;
    f1 = feval(func, samp.Y(idx,:)' + s);
    func_eval = func_eval + 1;
    
    % add the new point to the sample set
    [samp, idx1] = samp.addpoints(samp.Y(idx,:) + s', f1, delta, par);
    
    % calculate the ratio between real reduction in function value and 
    % the reduction in the value of the approximating model
    rho  = (samp.f(idx) - samp.f(idx1)) / (samp.f(idx) - m1);

    % evaluate the quality of the new point and update the trust region 
    [success, delta] = trust_region_update(delta, rho, samp.mgood, par);

	% Go to the new point if the step is successful. 
    if success
        iter_suc = iter_suc + 1;
        idx = idx1;
        samp = samp.centering(idx);
    end
    
	% Print iteration report.
    if par.print >= 2
        fprintf('| %5d |    %2d   | %+13.8e | %+5.2e | %+5.2e |  %5d  |\n',...
            iter, success, samp.f(idx), delta, rho, samp.m);
    end
    
    % Check a stopping criterion.
    if delta <= par.stop.tol_delta
        % Stop if the trust radius is too small.
        fprintf('Existing due to small trust region radius. \n')
        break; 
    end
        
	% Discard some points. 
    [samp, newindeces] = samp.deletepoints(delta, par);
    idx = newindeces(idx);
    
    % ask for poisedness improvement points 
    if par.algorithm.poised && samp.m < (n+1)*(n+2)/2
        points = samp.poisedness_improvement_points(1,delta, par);
        
        fvalues = nan(size(points,1),1);
        for i = 1:size(points,1)
            fvalues(i) = feval(func, points(i,:)');
        end
        func_eval = func_eval + size(points,1);
        
        samp = samp.addpoints(points, fvalues, delta, par);
    end
end  

x = samp.Y(idx,:)';
f = samp.f(idx);

%%
time = etime(clock,time);

if par.print >= 1
    % Print final report.
    fid = 1;
    %frewind(fid);
    fprintf(fid, '\nFinal Report for: %s (n=%2d) \n', func,n);
    fprintf(fid, 'Elapsed Time = %10.3e \n', time);
    fprintf(fid, 'Norm of Model gradient = %13.8e \n', norm(model.coefficients.g));
    fprintf(fid,'| #iter | #isuc | #fevals |  final f_value  | final tr_radius  |\n');
    fprintf(fid, '| %5d | %5d |  %5d  | %+13.8e | %+13.8e | \n',...
                  iter, iter_suc, func_eval, f, delta);
    fprintf('\n');
end

% End of dfo_tr.
