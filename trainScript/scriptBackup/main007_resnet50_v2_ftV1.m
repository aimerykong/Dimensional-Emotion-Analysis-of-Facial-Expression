%% add path and setup configuration
clc; clear; close all;

addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

%% prepare data
path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';
gpuId = 2;
gpuDevice(gpuId);

load(path_to_imdb) ;
meanVal = imdb.meta.mean_value;
% imdb.train.annot = (imdb.train.annot*9+1)/10;
% imdb.val.annot = (imdb.val.annot*9+1)/10;
% imdb.test.annot = (imdb.test.annot*9+1)/10;

imdb.meta.imagesize = [48,48,1];
%% configuration 
totalEpoch = 500;
learningRate = 1:totalEpoch;
learningRate = (1e-3) * (1-learningRate/totalEpoch).^0.9;
weightDecay=0.0005; % weightDecay: usually use the default value
%% initialize the model
saveFolder = 'main007_resnet50_v1';
modelName = 'DimEmotion_resnet_L1_net-epoch-940.mat';
netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

% netbasemodel = load('/home/skong2/projects/basemodels/imagenet-resnet-50-dag.mat');

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.meta.normalization.averageImage = meanVal; % imagenet mean values


fc1_f = netbasemodel.params(netbasemodel.getParamIndex('fc1_f')).value;
fc1_b = netbasemodel.params(netbasemodel.getParamIndex('fc1_b')).value;

netbasemodel.removeLayer('loss_L2');
netbasemodel.removeLayer('loss_L1');
netbasemodel.removeLayer('sigmoidOutput');
netbasemodel.removeLayer('fc1');


sName = 'pool5';
lName = 'res6_dropout' ;
block = dagnn.DropOut('rate', 0.5);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


lName = 'fc1';
netbasemodel.addLayer(lName , ...
    dagnn.Conv('size', [1 1 2048 3]), ...
    sName, lName, {'fc1_f', 'fc1_b'}) ;
ind = netbasemodel.getParamIndex('fc1_f');
% weights = randn(1, 1, 2048, 3, 'single')*sqrt(2/3);
netbasemodel.params(ind).value = fc1_f;
ind = netbasemodel.getParamIndex('fc1_b');
% weights = zeros(1, 3, 'single'); 
netbasemodel.params(ind).value = fc1_b;
sName = lName;


lName = 'sigmoidOutput';
netbasemodel.addLayer(lName, ...
    dagnn.Sigmoid, {sName}, lName);
lName = sName;


lossCellList = {'loss_L1', 2, 'loss_L2', 1};

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


netbasemodel.params(netbasemodel.getParamIndex('fc1_f')).learningRate = 4;
netbasemodel.params(netbasemodel.getParamIndex('fc1_b')).learningRate = 4;

RFinfo = netbasemodel.getVarReceptiveFields('data');
for i = 1:numel(netbasemodel.params)
    fprintf('%d\t%25s, \t%.2f',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);
    fprintf('\tsize: %dx%dx%dx%d\n', size(netbasemodel.params(i).value,1), size(netbasemodel.params(i).value,2), size(netbasemodel.params(i).value,3), size(netbasemodel.params(i).value,4));
end
%%
netbasemodel.meta.trainOpts.batchSize = 1 ; 
netbasemodel.meta.normalization.averageImage = single(meanVal);
netbasemodel.meta.normalization.imageSize = [];
netbasemodel.meta.trainOpts.learningRate = learningRate;
netbasemodel.meta.trainOpts.numEpochs = numel(learningRate);
%% modify the pre-trained model to fit the current size/problem/dataset/architecture, excluding the final layer
batchSize = 128; 
mopts.classifyType='L1'; 

% some parameters should be tuned
opts.batchSize = batchSize;
opts.learningRate = netbasemodel.meta.trainOpts.learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize = opts.batchSize;
%% setup to train network
opts.expDir = fullfile('./exp', 'main007_resnet50_v2_ftV1');
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

fn = getBatchWrapper_DimEmotion_stage1(bopts) ;

rng('default');
opts.gpus = gpuId ;
opts.checkpointFn = [];
opts.backPropAboveLayerName = 'conv1';

[netbasemodel, info] = cnntrainDag_DimEmotion_ResNet(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', lossCellList, opts);


