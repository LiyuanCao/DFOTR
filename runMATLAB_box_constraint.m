clear; clc; 
% Runs DFO for solving noisy problems

addpath('MATLAB_box_constraint')
% rng(2);
warning('off', 'MATLAB:nearlySingularMatrix')
% warning('on','all')

% Define function handle ((application, user provided))
obj_func = @arwhead;

% Initiate starting point
n = 50; 
% x0 = rand(100,1);
x0 = 0.6*ones(n,1);
lb = 0.5*ones(n,1);
ub = 0.95*ones(n,1);

% options
options.alg_model = 'quadratic';
options.alg_TR = 'box'; 
options.alg_TRsub = 'exact';
options.tr_delta = 0.1;
options.sample_max = 52;
options.verbose = 2;

% Call DFO-TR directly. 
% [x,f,info] = dfo_tr.optimize(obj_func, x0, lb, ub, options);

% Call DFO-TR with ask and tell framework. 
optimizer = dfo_tr(x0, lb, ub, options); 
while true
    x = optimizer.ask();
    fx = obj_func(x);
    optimizer.tell(x,fx);
    if optimizer.stop()
        break;
    end
end
x = optimizer.info.bestx;

% print result
fprintf('x = \n')
fprintf('%12.9f  %12.9f  %12.9f  %12.9f\n', x)
fprintf('\n')

rmpath('MATLAB_box_constraint')