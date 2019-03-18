function [f] = bdqrtic(x)
%
% Purpose:
%
%    Function bdqrtic is the Test Problem 61 in the 
%    Appendix of Conn et al (1994) and computes
%    the value of the objective function bdqrtic.
%
%    dim >= 5
%    Suggested initial point for optimization:
%    x = ones(10,1) or x = ones(20,1)
%    Minimum value: 0          if dim = 5000
%                   3.98382E+3 if dim = 1000
%                   1.98101E+3 if dim = 500
%                   3.78769E+2 if dim = 100
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
A   = (x(1:dim-4).^2 + 2*x(2:dim-3).^2 + 3*x(3:dim-2).^2 +...
      4*x(4:dim-1).^2 + 5*x(dim).^2).^2 - 4*x(1:dim-4)+3;
f   = sum(A);
%
% End of bdqrtic.
