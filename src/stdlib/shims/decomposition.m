classdef decomposition
    properties
        A
        checkCondition
        isIllCond
    end
    methods
        function obj = decomposition(A, varargin)
            obj.A = A;
            obj.checkCondition = true;
            for i = 1:2:length(varargin)
                if strcmpi(varargin{i}, 'CheckCondition')
                    obj.checkCondition = varargin{i+1};
                end
            end
            if obj.checkCondition
                obj.isIllCond = (rcond(A) < eps);
            else
                obj.isIllCond = false;
            end
        end
        function x = mldivide(obj, b)
            x = obj.A \ b;
        end
    end
end
