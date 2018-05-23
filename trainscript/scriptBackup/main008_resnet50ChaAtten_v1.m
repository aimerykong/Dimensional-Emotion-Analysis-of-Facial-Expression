%% add path and setup configuration
clc; clear; close all;

addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

gpuId = 3;
gpuDevice(gpuId);
%% prepare data
path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';

load(path_to_imdb) ;
%imdb.meta.mean_value = reshape([123.68, 116.779,  103.939],[1,1,3]); 
meanVal = imdb.meta.mean_value;
imdb.meta.imagesize = [48,48,3];
%% configuration 
totalEpoch = 1000;
learningRate = 1:totalEpoch;
learningRate = (1e-3) * (1-learningRate/totalEpoch).^0.9;
weightDecay=0.0005; % weightDecay: usually use the default value
%% initialize the model

netbasemodel = load('init_resnet50_ChaAtten.mat');
% netbasemodel = load('/home/skong2/projects/MarkovCNN/NYUv2_seg/initResNet50_ChaAtten.mat');

% netbasemodel = load(fullfile('/home/skong2/projects/MarkovCNN/NYUv2_seg/initADE20k_Res5ScaleAttention_ChaAtten.mat'));
% for i = 1:length(netbasemodel.layers)
%     if isfield(netbasemodel.layers(i).block, 'bnorm_moment_type_trn')
%         netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'noise_param_idx');
%         netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'noise_cache_size');
%         netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'bnorm_moment_type_trn');
%         netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'bnorm_moment_type_tst');
%     end
% end

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.meta.normalization.averageImage = reshape([123.68, 116.779,  103.939],[1,1,3]); 
netbasemodel.meta.inputSize = [48, 48, 3]; % imagenet mean values

%{
% netbasemodel.removeLayer('obj_div1_seg');
% netbasemodel.removeLayer('res8_interp');
% netbasemodel.removeLayer('res8_conv');
% netbasemodel.removeLayer('res7_dropout');
% netbasemodel.removeLayer('res7_relu');
% netbasemodel.removeLayer('res7_ChaAtten');
% netbasemodel.removeLayer('res7_conv');
% netbasemodel.removeLayer('res6_relu');
% netbasemodel.removeLayer('res6_ChaAtten');
% netbasemodel.removeLayer('res6_conv');

% netbasemodel.layers(netbasemodel.getLayerIndex('conv1')).block.stride = [1 1];
% netbasemodel.removeLayer('pool1');
% netbasemodel.setLayerInputs('res2_1_projBranch', {'conv1_relu'});
% netbasemodel.setLayerInputs('res2_1_1conv', {'conv1_relu'});


% ResBlock4
% netbasemodel.layers(55).block.stride = [1 1]; % res4_1_projBranch
% netbasemodel.layers(56).block.stride = [1 1]; % res4_1_1conv
netbasemodel.layers(85).block.stride = [2 2]; % res4_1_projBranch
netbasemodel.layers(87).block.stride = [2 2]; % res4_1_1conv
dilationRate = 1;
for idx = 1:length(netbasemodel.layers)
    if ~isempty(strfind(netbasemodel.layers(idx).name, 'res4')) ... 
            && isempty(strfind(netbasemodel.layers(idx).name, 'ScaleAttention')) ...  
            && strcmpi( class(netbasemodel.layers(idx).block), 'dagnn.Conv' ) ...  
            && isempty(strfind(netbasemodel.layers(idx).name, 'relu')) ...          
            && netbasemodel.layers(idx).block.size(1) == 3  
        netbasemodel.layers(idx).block.dilate = [dilationRate dilationRate];
        netbasemodel.layers(idx).block.pad = [dilationRate dilationRate];
        disp(netbasemodel.layers(idx).name);
    end
end

% ResBlock5
netbasemodel.layers(147).block.stride = [2 2]; % res5_1_projBranch
netbasemodel.layers(149).block.stride = [2 2]; % res5_1_1conv
dilationRate = 1;
for idx = 1:numel(netbasemodel.layers)    
    if ~isempty(strfind(netbasemodel.layers(idx).name, 'res5')) ...       
            && isempty(strfind(netbasemodel.layers(idx).name, 'ScaleAttention')) ...       
            && strcmpi( class(netbasemodel.layers(idx).block), 'dagnn.Conv' ) ...  
            && isempty(strfind(netbasemodel.layers(idx).name, 'relu')) ...
            && netbasemodel.layers(idx).block.size(1) == 3   
        netbasemodel.layers(idx).block.dilate = [dilationRate dilationRate];
        netbasemodel.layers(idx).block.pad = [dilationRate dilationRate];
        disp(netbasemodel.layers(idx).name);
    end
end
%}

lName = 'res6_pool';
netbasemodel.addLayer('res6_pool', dagnn.Pooling('poolSize', [6 6], ...
    'stride', 1, 'pad', 0, 'method', 'avg'), ...
    'res5_3_relu', ...
    'res6_pool') ;
sName = lName;

lName = 'res6_dropout' ;
block = dagnn.DropOut('rate', 0.5);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;

lName = 'res6_conv';
dimInput = 2048;
dimOutput = 3;
netbasemodel.addLayer(lName , ...
    dagnn.Conv('size', [1 1 dimInput 3]), ...
    sName, lName, {'res6_conv_f', 'res6_conv_b'}) ;
ind = netbasemodel.getParamIndex('res6_conv_f');
weights = randn(1, 1, dimInput, dimOutput, 'single')*sqrt(2/dimOutput);
netbasemodel.params(ind).value = weights;
ind = netbasemodel.getParamIndex('res6_conv_b');
weights = zeros(1, dimOutput, 'single'); 
netbasemodel.params(ind).value = weights;
sName = lName;

% lName = 'res6_relu';
% block = dagnn.ReLU('leak', 0);
% netbasemodel.addLayer(lName, block, sName, lName);
% sName = lName;
% 
% 
% lName = 'res7_dropout' ;
% block = dagnn.DropOut('rate', 0.5);
% netbasemodel.addLayer(lName, block, sName, lName);
% sName = lName;
% 
% 
% lName = 'res7_conv';
% netbasemodel.addLayer(lName , ...
%     dagnn.Conv('size', [1 1 128 3]), ...
%     sName, lName, {'res7_conv_f', 'res7_conv_b'}) ;
% ind = netbasemodel.getParamIndex('res7_conv_f');
% weights = randn(1, 1, 128, 3, 'single')*sqrt(2/3);
% netbasemodel.params(ind).value = weights;
% ind = netbasemodel.getParamIndex('res7_conv_b');
% weights = zeros(1, 3, 'single'); 
% netbasemodel.params(ind).value = weights;
% sName = lName;


% lName = 'sigmoidOutput';
% netbasemodel.addLayer(lName, ...
%     dagnn.Sigmoid, {sName}, lName);
% sName = lName;


lossCellList = {'loss_L1', 1, 'loss_L2', 1};

obj_name = sprintf('loss_L1');
gt_name =  sprintf('label');
netbasemodel.addLayer(obj_name, ...
    DimEmotionLoss('loss', 'reg_L1'), ... 
    {sName, gt_name}, obj_name);

gt_name =  'label';
obj_name = 'loss_L2';
netbasemodel.addLayer(obj_name, ...
    DimEmotionLoss('loss', 'reg_L2'), ...
    {sName, gt_name}, obj_name);

%% learning rate
for ii = 1:numel(netbasemodel.layers)
    curLayerName = netbasemodel.layers(ii).name;
    if ~isempty(strfind(curLayerName, 'relu')) || ~isempty(strfind(curLayerName, 'pool'))
        continue;
    elseif ~isempty(strfind(curLayerName, 'bn'))
        fprintf('\t%03d, %s\n', ii, curLayerName);
        netbasemodel.layers(ii).block.usingGlobal = 1;
        netbasemodel.params(netbasemodel.layers(ii).paramIndexes(1)).learningRate = 1.0;
        netbasemodel.params(netbasemodel.layers(ii).paramIndexes(2)).learningRate = 1.0;
        netbasemodel.params(netbasemodel.layers(ii).paramIndexes(3)).learningRate = 0.01;
        
        tmp = netbasemodel.params(netbasemodel.layers(ii).paramIndexes(1)).value;
        netbasemodel.params(netbasemodel.layers(ii).paramIndexes(1)).value = ones(size(tmp,1), 1, 'single'); % slope
        netbasemodel.params(netbasemodel.layers(ii).paramIndexes(2)).value = zeros(size(tmp,1), 1, 'single');  % bias
        netbasemodel.params(netbasemodel.layers(ii).paramIndexes(3)).value = zeros(size(tmp,1), 2, 'single'); % moments                
    elseif ~isempty(strfind(curLayerName, 'conv')) ||  (~isempty(strfind(curLayerName, 'res')) && isempty(strfind(curLayerName, 'relu')))
        fprintf('%03d, %s\n', ii, curLayerName);
        for paramIdx = netbasemodel.layers(ii).paramIndexes
            tmp = netbasemodel.params(paramIdx).value;
            netbasemodel.params(paramIdx).learningRate = 1.0;
            netbasemodel.params(paramIdx).value = randn(size(tmp,1), size(tmp,2), size(tmp,3), size(tmp,4), 'single')*sqrt(2/size(tmp,4));
        end
    else
        continue;
    end
end

% netbasemodel.params(netbasemodel.getParamIndex('fc1_f')).learningRate = 10;
netbasemodel.params(netbasemodel.getParamIndex('res6_conv_f')).learningRate = 10;
netbasemodel.params(netbasemodel.getParamIndex('res6_conv_b')).learningRate = 10;
% netbasemodel.params(netbasemodel.getParamIndex('res7_conv_f')).learningRate = 10;
% netbasemodel.params(netbasemodel.getParamIndex('res7_conv_b')).learningRate = 10;

RFinfo = netbasemodel.getVarReceptiveFields('data');
for i = 1:numel(netbasemodel.params)
    fprintf('%d\t%25s, \t%.2f',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);
    fprintf('\tsize: %dx%dx%dx%d\n', size(netbasemodel.params(i).value,1), size(netbasemodel.params(i).value,2), size(netbasemodel.params(i).value,3), size(netbasemodel.params(i).value,4));
end
%%
% showDagNetFlow(netbasemodel); % show information flow within the architecture
% net.print('Format', 'latex');
% netbasemodel.print({'image', [1280 1280 1 1]}, 'Format', 'dot', 'All', true);

netbasemodel.meta.trainOpts.batchSize = 1 ; 
netbasemodel.meta.normalization.averageImage = single(meanVal);
netbasemodel.meta.normalization.imageSize = [];
netbasemodel.meta.trainOpts.learningRate = learningRate;
netbasemodel.meta.trainOpts.numEpochs = numel(learningRate);
%% modify the pre-trained model to fit the current size/problem/dataset/architecture, excluding the final layer
batchSize = 10; % each batch contains one image, around 640x768-pixel resolution
mopts.classifyType='L1'; 

% some parameters should be tuned
opts.batchSize = batchSize;
opts.learningRate = netbasemodel.meta.trainOpts.learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize = opts.batchSize;
%% setup to train network
opts.expDir = fullfile('./exp', 'main008_resnet50ChaAtten_v1');
if ~isdir(opts.expDir)
    mkdir(opts.expDir);
end
opts.numSubBatches = 1 ;
opts.continue = true ;
opts.gpus = gpuId ;
%gpuDevice(opts.train.gpus); % don't want clear the memory
opts.prefetch = true ;
opts.sync = false ; % for speed
opts.cudnn = true ; % for speed
opts.learningRate = learningRate;
opts.numEpochs = numel(opts.learningRate);

% in case some dataset only has val/test
opts.val = imdb.val;
opts.train = imdb.train;

bopts = netbasemodel.meta.normalization;
bopts.imdb = imdb;
% bopts.numThreads = 12;

opts.train.backPropDepth = inf; % could limit the backprop
%% train
netbasemodelName = 'resnet';
prefixStr = [dataset, '_', netbasemodelName, '_', mopts.classifyType, '_'];

fn = getBatchWrapper_DimEmotion(bopts) ;

rng('default');
opts.checkpointFn = [];
opts.backPropAboveLayerName = 'conv1';

[netbasemodel, info] = cnntrainDag_DimEmotion_ResNet(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', lossCellList, opts);


