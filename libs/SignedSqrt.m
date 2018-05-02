classdef SignedSqrt < dagnn.ElementWise
    methods        
        function outputs = forward(obj, inputs, params)           
            outputs{1}=sign(inputs{1}).*sqrt(abs(inputs{1}));
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % should always set the ep LARGE enough to avoid gradient explosion
            ep=1e-1;            
            derInputs{1} = derOutputs{1} .* 0.5 ./ (sqrt(abs(inputs{1})) + ep);            
            derParams = {} ;            
        end
        
        function obj = SignedSqrt(varargin)
            obj.load(varargin) ;
        end
    end
end
