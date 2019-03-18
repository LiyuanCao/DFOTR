% function [success,delta,rho] = trust_region_update(f,m1,f1,delta,m,param) 
% 
% % Calculate the ratio between real reduction in function value and 
% % the reduction in the value of the approximating model
% rho  = (f - f1) / (f - m1);
% 
% %   Updating iterate and trust-region radius.
% if rho >= param.tr.eta0
%     % When the approximation is sufficiently good, 
%     % accept the new point.
%     success = 1;
% 
%     % When the approximation is very good, increase TR radius
%     if rho >= param.tr.eta2
%         delta = min(param.tr.gamma2*delta, 10^3);
%     end
% else
%     % When the approximation is bad, function value is not 
%     % significantly reduced, reject the new point.
%     success = 0;
%     
%     %reduce TR radius
%     if m > param.sample.min + 20
%         delta = param.tr.gamma1*delta;
%     end
% end

function  [success, delta, par] = trust_region_update(delta, rho, mgood, par)

    % Updating iterate and trust-region radius.
    if rho >= par.tr.toaccept
        % When the approximation is sufficiently good, accept the new point.
        success = 1;

        % When the approximation is very good, increase TR radius. 
        if rho >= par.tr.toexpand
            delta = min(par.tr.expand * delta, 1e3);
        end

	else
        % When the approximation is bad, function value is not 
        % significantly reduced, reject the new point.
        success = 0;
    
        % reduce TR radius
        if mgood >= par.tr.toshrink 
            delta = delta * par.tr.shrink;
        end
    end