function param = initialize_parameters(param, n)

param.n = n;

%% algorithm parameters
if ~isfield(param,'algorithm')
    param.algorithm = []; 
end

if ~isfield(param.algorithm,'model')
    param.algorithm.model = 'quadratic'; 
end

if ~isfield(param.algorithm,'tr')
    param.algorithm.tr = 'exact'; 
end

if ~isfield(param.algorithm,'poised')
    param.algorithm.poised = false; 
end

%% Sample Parameters
if ~isfield(param,'sample')
    param.sample = []; 
end

% initial size of the sample:
if ~isfield(param.sample, 'initial')
	param.sample.initial = n+1;
end

% minimum size of the sample:
% just enough points for linear interpolation.
if ~isfield(param.sample, 'min')
	param.sample.min = n + 1;
end

% maximum size of the sample:
% just enough points for quadratic interpolation.
if ~isfield(param.sample, 'max')
    param.sample.max = (n+1)*(n+2)/2; 
end

% How far is a sample point from the center considered too far?
% When the distance is >= delta * par.sample.toremove. 
if ~isfield(param.sample, 'toremove')
    param.sample.toremove = 50;
end

%% Trust Region Parameters
if ~isfield(param,'tr')
    param.tr = []; 
end

% Initial trust radius
if ~isfield(param.tr,'delta')
    param.tr.delta = 1;   
end

% Ratio ared/pred for an iterate to be considered successful.
if ~isfield(param.tr, 'toaccept')
    param.tr.toaccept = eps;    
end

% Ratio ared/pred for contracting the trust
% radius if want to make it higher than eta0. 
% good for fatser stopping, 
% not good for noisy situations. 
% eta1       = 0.25;    

% Ratio ared/pred for expanding the trust radius.
if ~isfield(param.tr, 'toexpand')
    param.tr.toexpand = 0.5;
end

% Contraction parameter for the trust radius - 
% This is a very slow down to avoid getting stuck in a noisy zone
if ~isfield(param.tr, 'shrink')
	param.tr.shrink = 0.5^(10 / (10+n)); 
end

% Expansion parameter for the trust radius.
if ~isfield(param.tr, 'expand')
    param.tr.expand = 1.5;
end

% minimum number of sample points required to start shrink the trust region
if ~isfield(param.tr, 'toshrink')
    param.tr.toshrink = min(ceil(param.sample.min * 1.05), param.sample.max);
end

%% Stopping Criteria Parameters
if ~isfield(param,'stop')
    param.stop = []; 
end

% maximum number of iteration
if ~isfield(param.stop, 'iter')
	param.stop.iter  = 2e3;
end

% a small trust region radius
if ~isfield(param.stop, 'tol_delta')
	param.stop.tol_delta  = 1e-5;
end

% Toleracy for considering function improvement too small.  
if ~isfield(param.stop, 'tol_f')
    param.stop.tol_f = 1e-7;
end

% a small gradient
% if ~isfield(param.stop, 'tol_g')
%     param.stop.tol_g = 10^-5;
% end

%% print
if ~isfield(param,'print')
    param.print = 2; 
end
