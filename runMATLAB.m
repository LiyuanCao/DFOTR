clear; clc; 
% Runs DFO for solving noisy problems

addpath('matlab')
% rng(2);
warning('off', 'MATLAB:nearlySingularMatrix')
% warning('on','all')

% Define function handle ((application, user provided))
obj_func = @arwhead;

% Initiate starting point
% x0 = rand(100,1);
x0 = 0.5*ones(50,1);

% options
options.alg_model = 'quadratic';
options.alg_TRsub = 'exact';
options.tr_delta = 0.1;
options.sample_min = length(x0) + 1;
options.verbose = 2;

% Call DFO-TR directly. 
% [x,f,info] = dfo_tr.optimize(obj_func,x0, options);

% Call DFO-TR with ask and tell framework. 
optimizer = dfo_tr(x0, options); 
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

rmpath('matlab')