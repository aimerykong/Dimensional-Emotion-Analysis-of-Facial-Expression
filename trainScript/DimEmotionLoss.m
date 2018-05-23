classdef DimEmotionLoss < dagnn.Loss
%     properties
%         ignoreAverage=0
%         normalise=1;
%     end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnloss_DimEmotion(inputs{1}, inputs{2}, [], ...
                'loss', obj.loss, ...
                'instanceWeights', 1) ;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)            
            derInputs{1} = vl_nnloss_DimEmotion(inputs{1}, inputs{2}, derOutputs{1}, ...
                'loss', obj.loss, ...
                'instanceWeights', 1) ;
            derInputs{2} = [] ;
            derParams = {} ;
        end

        function obj = DimEmotionLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
