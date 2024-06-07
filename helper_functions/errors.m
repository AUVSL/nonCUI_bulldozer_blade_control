function [rmse, me] = errors(x)
    [n,~] = size(x);
    rmse = sqrt(sum(x.^2)/n);
    me = max(abs(x));
end