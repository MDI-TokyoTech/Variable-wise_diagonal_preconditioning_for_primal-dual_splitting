function result = ProxL21norm(X, gamma)
    T = max(1 - gamma./sqrt(sum(X.*X, 2)), 0);
    result = T.*X;

end