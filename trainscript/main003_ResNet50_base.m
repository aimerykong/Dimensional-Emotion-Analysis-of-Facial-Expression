%% add path and setup configuration
clc; clear; close all;
imgFig = figure(1); set(imgFig, 'Position',[100,100,1100,500]); % [1 1 width height]

addpath('../libs/exportFig');
addpath('../libs/layerExt');
addpath('../libs/myFunctions');
path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

gpuId = 4;
gpuDevice(gpuId);
%% prepare data
path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';
load(path_to_imdb) ;
%imdb.meta.mean_value = reshape([123.68, 116.779,  103.939],[1,1,3]); %imagenet
% imdb.meta.mean_value = reshape([129.1863, 104.7624, 93.5940],[1,1,3]); %face
imdb.meta.mean_value = reshape([131.0912 103.8827 91.4953],[1,1,3]); %vggface2

meanVal = imdb.meta.mean_value;
imdb.meta.imagesize = [224,224,3];
imdb.train.annot = imdb.train.annot*9;
imdb.val.annot = imdb.val.annot*9;
imdb.test.annot = imdb.test.annot*9;
%% configuration 
batchSize = 50; 
totalEpoch = 500;
learningRate = 1:totalEpoch;
learningRate = (1e-5) * (1-learningRate/totalEpoch).^0.9;
weightDecay=0.0005; % weightDecay: usually use the default value
%% initialize the model
saveFolder = 'main012_vggface2_v1_basemodel';
modelName = 'DimEmotion_resnet_L1_net-epoch-492.mat'; % 

netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = dagnn.DagNN.loadobj(netbasemodel.net);

netbasemodel.meta.normalization.averageImage = meanVal; 
netbasemodel.meta.inputSize = imdb.meta.imagesize; % imagenet mean values
%% modify model architecture
lossCellList = {'loss_L1', 1, 'loss_L2', 1};
%% learning rate
netbasemodel.params(netbasemodel.getParamIndex('block6_conv_f')).learningRate = 1;
netbasemodel.params(netbasemodel.getParamIndex('block6_conv_b')).learningRate = 1;

RFinfo = netbasemodel.getVarReceptiveFields('data');
for i = 1:numel(netbasemodel.params)
    fprintf('%d\t%25s, \t%.2f',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);
    fprintf('\tsize: %dx%dx%dx%d\n', size(netbasemodel.params(i).value,1), size(netbasemodel.params(i).value,2), size(netbasemodel.params(i).value,3), size(netbasemodel.params(i).value,4));
end

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
opts.expDir = fullfile('./exp', 'main012_vggface2_v3_ftv1_1topConv');
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
opts.backPropAboveLayerName = 'conv1_7x7_s2';

[netbasemodel, info] = cnntrainDag_DimEmotion_ResNet(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', lossCellList, opts);


