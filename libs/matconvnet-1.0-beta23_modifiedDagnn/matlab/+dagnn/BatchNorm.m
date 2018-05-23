classdef BatchNorm < dagnn.ElementWise
    properties
        numChannels
        epsilon = 1e-5
        opts = {'NoCuDNN'} % ours seems slightly faster
        usingGlobal = false;
    end
    
    properties (Transient)
        moments
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            if strcmpi(obj.net.mode, 'test')
                outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'moments', params{3}, 'epsilon', obj.epsilon, obj.opts{:}) ;
            elseif strcmpi(obj.net.mode, 'trainGlobalBN') && obj.usingGlobal
                [outputs{1},obj.moments] = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'moments', params{3}, 'epsilon', obj.epsilon, obj.opts{:}) ;                
            elseif strcmpi(obj.net.mode, 'trainGlobalBN') && ~obj.usingGlobal
                [outputs{1},obj.moments] = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'epsilon', obj.epsilon, obj.opts{:}) ;
            elseif strcmpi(obj.net.mode, 'trainLocalBN') && obj.usingGlobal
                [outputs{1},obj.moments] = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'moments', params{3}, 'epsilon', obj.epsilon, obj.opts{:}) ;
            elseif strcmpi(obj.net.mode, 'trainLocalBN') && ~obj.usingGlobal
                [outputs{1},obj.moments] = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'epsilon', obj.epsilon, obj.opts{:}) ;
                
            elseif strcmpi(obj.net.mode, 'normal') && obj.usingGlobal
                outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'moments', params{3}, 'epsilon', obj.epsilon, obj.opts{:}) ;
                
%                 X = inputs{1};
%                 mu = params{3}(:,1);
%                 sigma = params{3}(:,2);
%                 X = bsxfun(@minus, X, reshape(mu, [1, 1, numel(mu)]) );
%                 Y =  bsxfun(@times, X, reshape(params{1}./sigma,[1,1,numel(mu)]));
%                 Y = bsxfun(@plus, Y, reshape(params{2},[1,1,numel(mu)]));
                
            elseif strcmpi(obj.net.mode, 'normal') && ~obj.usingGlobal
                [outputs{1},obj.moments] = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'epsilon', obj.epsilon, obj.opts{:}) ;
            else
                [outputs{1},obj.moments] = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'epsilon', obj.epsilon, obj.opts{:}) ;
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if strcmpi(obj.net.mode, 'normal') && obj.usingGlobal
                [derInputs{1}, derParams{1}, derParams{2}, derParams{3}] = ...
                    vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                    'epsilon', obj.epsilon, ...
                    'moments', params{3}, ...
                    obj.opts{:}) ;
                derParams{1} = derParams{1}*0;
                derParams{2} = derParams{2}*0;
                derParams{3} = derParams{3}*0;
                obj.moments = [] ;
            else
                [derInputs{1}, derParams{1}, derParams{2}, derParams{3}] = ...
                    vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                    'epsilon', obj.epsilon, ...
                    'moments', obj.moments, ...
                    obj.opts{:}) ;
                obj.moments = [] ;
                % multiply the moments update by the number of images in the batch
                % this is required to make the update additive for subbatches
                % and will eventually be normalized away
                derParams{3} = derParams{3} * size(inputs{1},4) ;
            end
            
        end
        
        % ---------------------------------------------------------------------
        function obj = BatchNorm(varargin)
            obj.load(varargin{:}) ;
        end
        
        function params = initParams(obj)
            params{1} = ones(obj.numChannels,1,'single') ;
            params{2} = zeros(obj.numChannels,1,'single') ;
            params{3} = zeros(obj.numChannels,2,'single') ;
        end
        
        function attach(obj, net, index)
            attach@dagnn.ElementWise(obj, net, index) ;
            p = net.getParamIndex(net.layers(index).params{3}) ;
            net.params(p).trainMethod = 'average' ;
            net.params(p).learningRate = 0.1 ;
        end
    end
end
