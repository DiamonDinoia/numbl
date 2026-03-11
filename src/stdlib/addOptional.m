function addOptional(obj, name, default, validator)
    if nargin < 4
        obj.addOptional(name, default);
    else
        obj.addOptional(name, default, validator);
    end
end
