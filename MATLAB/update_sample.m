function [Y,f_values,Ynorms,m,L] = ...
    update_sample(x,x1,f1,Y,f_values,Ynorms,m,L,delta,iter,param)

% Add the new point into the sample set.
if m < param.sample.max
    % Append the new point to the end of the sample set.
    m = m + 1; 

    Y(:,m)      = x1;
    f_values(m) = f1;

    if all(x == x1)
        Ynorms = sqrt(sum( bsxfun(@minus, Y, x).^2 ));
    else
        Ynorms(m) = norm(x1-x);
    end
else
    % Replace the farthest point with the new point.
    [~, farthest] = max(Ynorms);

    Y(:,farthest)      = x1;
    f_values(farthest) = f1;
    if all(x == x1)
        Ynorms= sqrt(sum( bsxfun(@minus, Y, x).^2 ));
    else
        Ynorms(farthest) = norm(x1-x);
    end
    L = [];
end

% discard points that are too far away
if mod(iter,20) == 0
    % points to be deleted
    [~,idx] = sort(Ynorms);
    idx = idx(param.sample.min+1:end); % save minY points from deletion
    idx = intersect(idx, find(Ynorms>delta*100)); 

    % points to be saved
    temp = true(m,1);
    temp(idx) = false;

    Y        = Y(:, temp);
    f_values = f_values(temp);
    Ynorms   = Ynorms(temp);
    if m > size(Y,2)
        m       = size(Y, 2);
        L = [];
    end
end