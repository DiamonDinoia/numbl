function addParameter(obj, name, default, validator)
    if nargin < 4
        obj.addParameter(name, default);
    else
        obj.addParameter(name, default, validator);
    end
end
