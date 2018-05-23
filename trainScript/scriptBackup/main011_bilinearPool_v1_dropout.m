%% add path and setup configuration
clc; clear; close all;

imgFig = figure(1);
set(imgFig, 'Position',[100,100,1100,500]); % [1 1 width height]


addpath(genpath('./bcnn'));
addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

gpuId = 1;
gpuDevice(gpuId);
%% prepare data
path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';
load(path_to_imdb) ;
%imdb.meta.mean_value = reshape([123.68, 116.779,  103.939],[1,1,3]); %imagenet
imdb.meta.mean_value = reshape([129.1863, 104.7624, 93.5940],[1,1,3]); %face
meanVal = imdb.meta.mean_value;
imdb.meta.imagesize = [224,224,3];
imdb.train.annot = imdb.train.annot*9;
imdb.val.annot = imdb.val.annot*9;
imdb.test.annot = imdb.test.annot*9;
%% configuration 
batchSize = 50; 
totalEpoch = 500;
learningRate = 1:totalEpoch;
learningRate = (1e-4) * (1-learningRate/totalEpoch).^0.9;
weightDecay=0.0005; % weightDecay: usually use the default value
%% initialize the model
% modelPath = fullfile('/home/skong2/projects/basemodels','vgg-face.mat') ;
% netbasemodel = load(modelPath) ;
% netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
% netbasemodel = vl_simplenn_tidy(netbasemodel) ;

saveFolder = 'main010_basemodel_v2_train';
modelName = 'DimEmotion_resnet_L1_net-epoch-422.mat'; % 245
netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

netbasemodel.layers(38).block = rmfield(netbasemodel.layers(38).block,'ignoreAverage');
netbasemodel.layers(38).block = rmfield(netbasemodel.layers(38).block,'normalise');
netbasemodel.layers(39).block = rmfield(netbasemodel.layers(39).block,'ignoreAverage');
netbasemodel.layers(39).block = rmfield(netbasemodel.layers(39).block,'normalise');

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.meta.normalization.averageImage = meanVal; 
netbasemodel.meta.inputSize = imdb.meta.imagesize; % imagenet mean values
%% modify model architecture
% netbasemodel.setLayerInputs('conv1_1', {'data'})
netbasemodel.removeLayer('loss_L2');
netbasemodel.removeLayer('loss_L1');
netbasemodel.removeLayer('output');
netbasemodel.removeLayer('dropout7');
netbasemodel.removeLayer('relu7');
netbasemodel.removeLayer('fc7');
netbasemodel.removeLayer('relu6');
netbasemodel.removeLayer('fc6');
netbasemodel.removeLayer('pool5');

sName = 'x30';
lName = 'bilinearPoolLayer';
netbasemodel.addLayer(lName, BilinearPool(), sName, lName);
sName = lName;

lName = 'signedSqrtLayer';
netbasemodel.addLayer(lName, SignedSqrt(), sName, lName) ;
sName = lName;

lName = 'L2normLayer';
netbasemodel.addLayer(lName, L2normalization(), sName, lName) ;
sName = lName;

lName = 'dropout7' ;
block = dagnn.DropOut('rate', 0.5);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;

lName = 'output';
dimInput = 262144;
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
%{
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
%}
netbasemodel.params(netbasemodel.getParamIndex('res6_conv_f')).learningRate = 10;
netbasemodel.params(netbasemodel.getParamIndex('res6_conv_b')).learningRate = 20;

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
mopts.classifyType='L1'; 

% some parameters should be tuned
opts.batchSize = batchSize;
opts.learningRate = netbasemodel.meta.trainOpts.learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize = opts.batchSize;
%% setup to train network
opts.expDir = fullfile('./exp', 'main011_bilinearPool_v1_dropout');
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
opts.backPropAboveLayerName = 'conv1_1';

[netbasemodel, info] = cnntrainDag_DimEmotion_ResNet(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', lossCellList, opts);


