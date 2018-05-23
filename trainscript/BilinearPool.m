classdef BilinearPool < dagnn.ElementWise
    %     properties
    %         t=0.1
    %     end
    
    methods        
        function outputs = forward(obj, inputs, params)
            x = inputs{1};
            % [height width channels batchsize]
            [h, w, ch, bs] = size(x);
            gpuMode = isa(x, 'gpuArray');
            if gpuMode
                y = gpuArray(zeros([1, 1, ch*ch, bs], 'single'));
            else
                y = zeros([1, 1, ch*ch, bs], 'single');
            end
            for b = 1:bs
                a = reshape(x(:,:,:,b), [h*w, ch]);
                y(1,1,:, b) = reshape(a'*a, [1 ch*ch]);
            end            
            outputs{1} = y;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            x = inputs{1};
            [h, w, ch, bs] = size(x);
            dzdy = derOutputs{1};
            gpuMode = isa(x, 'gpuArray');
            if gpuMode
                y = gpuArray(zeros(size(x), 'single'));
            else
                y = zeros(size(x), 'single');
            end
            for b=1:bs
                dzdy_b = reshape(dzdy(1,1,:,b), [ch, ch]);
                a = reshape(x(:,:,:,b), [h*w, ch]);
                % yang: bug here, should be 2* the original
                y(:, :, :, b) = 2*reshape(a*dzdy_b, [h, w, ch]);
            end
            derInputs{1} = y;
            derParams = {} ;
        end
        
        function obj = BilinearPool(varargin)
            obj.load(varargin) ;
        end
    end
end
