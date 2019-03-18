function [f] = rosenbr(x);
%
% Purpose:
%
%    Function rosenbr is the Rosenbrock's function in Powell (1964) 
%    and computes the value of the objective function rosenbr.
%
%    dim = 2
%    Suggested initial point for optimization:
%    x = [-1.2 1]'
%    Minimum point: [1 1]'
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
f = 100*(x(1,1)^2 - x(2,1))^2 + (1-x(1,1))^2;
%
% End of rosenbr.
