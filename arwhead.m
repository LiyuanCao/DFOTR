function [f] = arwhead(x)
%
% Purpose:
%
%    Function arwhead is the Test Problem 55 in the 
%    Appendix of Conn et al (1994) and computes
%    the value of the objective function arwhead.
%
%    dim >= 2
%    Suggested initial point for optimization:
%    x = ones(10,1) or x = ones(20,1)
%    Minimum value: 0
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
dim = length(x);
A   = (x.^2 + x(dim,1)^2).^2 - 4*x+3;
f   = sum(A) - A(dim,1);
%
% End of arwhead.
