function result = MixedL12NormGraph(x, Group_index_matrix_func)
    result = sum(sqrt(Group_index_matrix_func*(x.^2)), "all");
end