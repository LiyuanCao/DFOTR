clear; clc; 


% rng(2);
warning('off', 'MATLAB:nearlySingularMatrix')
% warning('on','all')
rmpath('MATLAB')
rmpath('MATLAB_box_constraint')

% Define function handle ((application, user provided))
obj_func = @arwhead;

% Initiate starting point
n = 50; 
% x0 = rand(100,1);
x0 = 0.6*ones(n,1);
lb = 0.5*ones(n,1);
ub = 0.95*ones(n,1);

%% Call DFO-TR with box constraints. 
addpath('MATLAB_box_constraint')
options.alg_model = 'quadratic';
options.alg_TR = 'box'; 
options.tr_delta = 0.1;
options.verbose = 2;
[x,f,info] = dfo_tr.optimize(obj_func, x0, lb, ub, options)
rmpath('MATLAB_box_constraint')

%% Call DFO-TR with mapping to Rn
addpath('MATLAB')
options.alg_TR = 'ball'; 
encoding_func = @(x) (ub-lb).*(sin(x)/2 + 0.5) + lb;
encoded_obj = @(x) obj_func(encoding_func(x)); 
[x,f,info] = dfo_tr.optimize(encoded_obj, zeros(n,1), options)

rmpath('MATLAB')