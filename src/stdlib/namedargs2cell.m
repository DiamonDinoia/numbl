function C = namedargs2cell(S)
    names = fieldnames(S);
    n = numel(names);
    C = cell(1, 2 * n);
    for i = 1:n
        C{2*i - 1} = names{i};
        C{2*i} = S.(names{i});
    end
end
