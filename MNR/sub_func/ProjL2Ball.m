% this function is the proximal operation of
% \iota_{\|\cdot-V\|_{F}<=\epsilon}(X)
% input is a 3D tensor

function result = T_proj_l2ball(X, V, epsilon)
    l2distance = sqrt(sum((X - V).^2, "all"));
    if l2distance <= epsilon
        result = X;
    else
        result = V + epsilon*(X - V)/l2distance;
    end
end
