%% add path and setup configuration
clc; clear; close all;

imgFig = figure(1);
set(imgFig, 'Position',[100,100,1100,500]); % [1 1 width height]

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
saveFolder = 'main013_vggface2_bilinear_v1';
modelName = 'DimEmotion_resnet_L1_net-epoch-500.mat'; %
netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.meta.normalization.averageImage = meanVal; 
netbasemodel.meta.inputSize = imdb.meta.imagesize; % imagenet mean values
%% modify model architecture
netbasemodel.removeLayer('loss_L2');
netbasemodel.removeLayer('loss_L1');
netbasemodel.removeLayer('output');

sName = 'L2normLayer';
lName = 'fc1';
dimInput = 262144;
dimOutput = 512;
netbasemodel.addLayer(lName , ...
    dagnn.Conv('size', [1 1 dimInput dimOutput]), ...
    sName, lName, {'fc1_f', 'fc1_b'}) ;
ind = netbasemodel.getParamIndex('fc1_f');
weights = randn(1, 1, dimInput, dimOutput, 'single')*sqrt(2/dimOutput);
netbasemodel.params(ind).value = weights;
ind = netbasemodel.getParamIndex('fc1_b');
weights = zeros(1, dimOutput, 'single'); 
netbasemodel.params(ind).value = weights;
sName = lName;


lName = 'dropout1' ;
block = dagnn.DropOut('rate', 0.5);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;

lName = 'output';
dimInput = dimOutput;
dimOutput = 3;
netbasemodel.addLayer(lName , ...
    dagnn.Conv('size', [1 1 dimInput dimOutput]), ...
    sName, lName, {'output_f', 'output_b'}) ;
ind = netbasemodel.getParamIndex('output_f');
weights = randn(1, 1, dimInput, dimOutput, 'single')*sqrt(2/dimOutput);
netbasemodel.params(ind).value = weights;
ind = netbasemodel.getParamIndex('output_b');
weights = zeros(1, dimOutput, 'single'); 
netbasemodel.params(ind).value = weights;
sName = lName;


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
netbasemodel.params(netbasemodel.getParamIndex('fc1_f')).learningRate = 10;
netbasemodel.params(netbasemodel.getParamIndex('fc1_b')).learningRate = 20;
netbasemodel.params(netbasemodel.getParamIndex('output_f')).learningRate = 10;
netbasemodel.params(netbasemodel.getParamIndex('output_b')).learningRate = 20;

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
mopts.classifyType='L1'; 

% some parameters should be tuned
opts.batchSize = batchSize;
opts.learningRate = netbasemodel.meta.trainOpts.learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize = opts.batchSize;
%% setup to train network
opts.expDir = fullfile('./exp', 'main013_vggface2_bilinear_v2_ftV1_fc1');
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


