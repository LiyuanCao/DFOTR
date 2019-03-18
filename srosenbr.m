function [f] = srosenbr(x);
%
% Purpose:
%
%    Function srosenbr is the extended Rosenbrock's
%    function in Dennis and Schnabel (1996) and computes
%    the value of the objective function srosenbr.
%
%    dim >= 2 and multiple of 2
%    Suggested initial point for optimization:
%    x = repmat([-1.2 1],1,5)' or x = repmat([-1.2 1],1,10)'
%    Minimum point: [1 ... 1]'
%
% Input:  
%
%         x (point given by the optimizer).
%
% Output: 
%
%         f (function value at x).
%
% Written by A. L. Custodio and L. N. Vicente.
%
% Version April 2004.
%
%
dim  = floor(length(x)/2);
A    = 100*(x(1:2*dim-1).^2 - x(2:2*dim)).^2 + (1-x(1:2*dim-1)).^2;
mask = mod([1:2*dim-1],2)';
f    = sum(A(logical(mask)));
%
% End of srosenbr.
