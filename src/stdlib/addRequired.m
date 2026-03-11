function addRequired(obj, name, validator)
    if nargin < 3
        obj.addRequired(name);
    else
        obj.addRequired(name, validator);
    end
end
