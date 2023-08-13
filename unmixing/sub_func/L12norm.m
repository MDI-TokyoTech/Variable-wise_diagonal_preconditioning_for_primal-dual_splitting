function result = L12norm(X)
    result = sum(sqrt(sum(X.^2, 2)));
end