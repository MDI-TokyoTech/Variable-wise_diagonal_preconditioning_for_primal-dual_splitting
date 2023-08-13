% this function is the proximal operation of
% \iota_{\|\cdot-V\|_{F}<=\epsilon}(X)
% input is a 3D tensor

function result = ProjL2Ball(x, v, epsilon)
    l2distance = sqrt(sum((x - v).^2, "all"));
    if l2distance <= epsilon
        result = x;
    else
        result = v + epsilon*(x - v)/l2distance;
    end
end
