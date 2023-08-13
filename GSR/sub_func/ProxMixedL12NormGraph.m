function result = ProxMixedL12NormGraph(x, gamma, Group_index_matrix)
    result = max(1 - gamma./sqrt(Group_index_matrix*(x.^2)), 0).*x;
end
