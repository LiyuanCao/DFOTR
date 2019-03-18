clear; clc; 
% Runs DFO for solving noisy problems

addpath('matlab')

% Define function handle ((application, user provided))
% Initiate starting point

func_f = 'arwhead';
% rng(2);
% y = rand(100,1);
y = 0.5*ones(50,1);

% warning('off', 'MATLAB:nearlySingularMatrix')
warning('on','all')

% Call DFOTR
param = [];
% param.algorithm.model = 'quadratic';
% param.algorithm.poised = false;
% param.tr.delta = 0.1;
% param.stop.tol_delta = 1e-12;
% param.stop.tol_f = 1e-12;
% param.stop.iter = 1e5;
% param.sample.toremove = 30;
[x,f] = dfo_tr(func_f,y, param);

% print result
fprintf('x = \n')
fprintf('%12.9f  %12.9f  %12.9f  %12.9f\n', x)
fprintf('\n')

rmpath('matlab')