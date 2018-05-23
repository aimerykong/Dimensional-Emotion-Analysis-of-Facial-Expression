function [net,stats] = cnntrainDag_DimEmotion_ResNet(net, prefixStr, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.
%
% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%%
opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 1 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 200 ;
opts.learningRate = 0.000000001 ; % !!!!!!!!!!!!!!!
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.sync = false ; % for speed
opts.cudnn = true ; % for speed
opts.backPropDepth = inf; % could limit the backprop
opts.backPropAboveLayerName = 'conv1_1';

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.checkpointFn = []; % will be called after every epoch
opts.plotStatistics = true;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
%if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
%if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
%if isnan(opts.train), opts.train = [] ; end


%% Initialization
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    if isempty(opts.derOutputs)
        error('DEROUTPUTS must be specified when training.\n') ;
    end
end

state.getBatch = getBatch ;
stats = [] ;

%% Train and validate
modelPath = @(ep) fullfile(opts.expDir, sprintf('%snet-epoch-%d.mat', prefixStr, ep));
modelFigPath = fullfile(opts.expDir, [prefixStr 'net-train.pdf']) ;
start = opts.continue * myfindLastCheckpoint(opts.expDir, prefixStr) ;

if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    [net, stats] = loadState(modelPath(start)) ;
end

plotLearningCurves(stats);
for epoch=start+1:opts.numEpochs
    
    % Set the random seed based on the epoch and opts.randomSeed.
    % This is important for reproducibility, including when training
    % is restarted from a checkpoint.
    
    rng(epoch + opts.randomSeed) ;
    prepareGPUs(opts, epoch == start+1) ;
    
    % Train for one epoch.
    
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    
    state.train = randperm(size(imdb.train.image,4)) ; % shuffle
    state.val = randperm(size(imdb.val.image,4)) ;
    
    
    state.imdb = imdb ;
    
    if numel(opts.gpus) <= 1
        [stats.train(epoch),prof] = DimEmotion_ResNet_process_epoch(net, state, opts, 'train', opts.numEpochs) ;
        stats.val(epoch) = DimEmotion_ResNet_process_epoch(net, state, opts, 'val', opts.numEpochs) ;
        
        if opts.profile
            profview(0,prof) ;
            keyboard ;
        end
    else
        savedNet = net.saveobj() ;
        spmd
            net_ = dagnn.DagNN.loadobj(savedNet) ;
            [stats_.train, prof_] = DimEmotion_ResNet_process_epoch(net_, state, opts, 'train') ;
            stats_.val = DimEmotion_ResNet_process_epoch(net_, state, opts, 'val') ;
            if labindex == 1, savedNet_ = net_.saveobj() ; end
        end
        net = dagnn.DagNN.loadobj(savedNet_{1}) ;
        stats__ = accumulateStats(stats_) ;
        stats.train(epoch) = stats__.train ;
        stats.val(epoch) = stats__.val ;
        if opts.profile
            mpiprofile('viewer', [prof_{:,1}]) ;
            keyboard ;
        end
        clear net_ stats_ stats__ savedNet savedNet_ ;
    end
    
    % save
    %if ~evaluateMode
    saveState(modelPath(epoch), net, stats) ;
    %end
    
    if opts.plotStatistics
        switchFigure(1) ; clf ;
        plots = setdiff(...
            cat(2,...
            fieldnames(stats.train)', ...
            fieldnames(stats.val)'), {'num', 'time'}) ;
        for p = plots
            p = char(p) ;
            values = zeros(0, length(stats.train)) ;
            leg = {} ;
            for f = {'train', 'val'}
                f = char(f) ;
                if isfield(stats.(f), p)
                    tmp = [stats.(f).(p)] ;
                    values(end+1,:) = tmp(1,:)' ;
                    leg{end+1} = f ;
                end
            end
            subplot(1,numel(plots),find(strcmp(p,plots))) ;
            if size(values,2)>5
                plot(5:size(values,2), values(:,5:end)','o-') ; % don't plot the first epoch
                xlabel('epoch') ;
            else
                plot(1:size(values,2), values(:, 1:end)','o-') ; % don't plot the first epoch
                xlabel('epoch') ;                
            end
            
            [minVal,minIdx] = min(values(2,:));
            [minValTr,minIdxTr] = min(values(1,:));
            title(sprintf('%s tsErr%.4f (%d) trErr%.4f (%d) ', p, min(values(2,:)), minIdx, min(values(1,:)),minIdxTr)) ;
            legend(leg{:},'location', 'SouthOutside') ;
            grid on ;
        end
        drawnow ;
        [curpath, curname, curext] = fileparts(modelFigPath);
        export_fig(fullfile(curpath, [curname, '.png']), '-png');
    end
    if ~isempty(opts.checkpointFn),
        opts.checkpointFn();
    end
    
end


% -------------------------------------------------------------------------
% function unmap_gradients(mmap)
% -------------------------------------------------------------------------



