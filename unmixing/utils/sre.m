function result = sre(A, B)
result = 10*log10(sum(A.^2, "all")./sum((A - B).^2, "all"));
end

