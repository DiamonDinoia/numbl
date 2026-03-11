classdef DenseLU
    properties
        L
        U
        P
    end
    methods
        function obj = DenseLU(A)
            [obj.L, obj.U, obj.P] = lu(A);
        end
        function x = solve(obj, b, ~)
            x = obj.U \ (obj.L \ (obj.P * b));
        end
    end
end