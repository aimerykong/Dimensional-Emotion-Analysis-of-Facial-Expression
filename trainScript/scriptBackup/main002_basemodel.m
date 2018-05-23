% nohup matlab -nodesktop -nodisplay < main002_basemodel.m 1>main002_basemodel.log &
clear
close all
clc;
%% specify the path
path_to_matconvnet = '/home/skong/data/MarkovCNN/PSPNet/matconvnet-1.0-beta23_modifiedDagnn/';
path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';
gpuId = 1;
gpuDevice(gpuId);
%% prepare data
load(path_to_imdb) ;
meanVal = imdb.meta.mean_value;
imdb.meta.imagesize = [48,48,1];
%% import packages
addpath(genpath('/mnt/data2/skong/nicolette_singleworm_seg/binarySeg/exportFig'));
addpath(genpath('/mnt/data2/skong/nicolette_singleworm_seg/binarySeg/layerExt'));
addpath(genpath('/mnt/data2/skong/nicolette_singleworm_seg/binarySeg/myFunctions'));
addpath(genpath('/mnt/data2/skong/nicolette_singleworm_seg/binarySeg/init'));
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile(path_to_matconvnet,'examples')));
%% configuration 
totalEpoch = 1000;
learningRate = 1:totalEpoch;
learningRate = (2e-3) * (1-learningRate/totalEpoch).^0.9;
weightDecay=0.0005; % weightDecay: usually use the default value
%% initialize the model
netbasemodelName = 'resnet';
depth = 40;
netbasemodel = resnet_init_forDimEmotion(depth, 'nClasses', 128, ...
                        'batchNormalization', 1, ...
                        'networkType', 'resnet') ;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
% netbasemodel.removeLayer('error5'); % remove layer
% netbasemodel.removeLayer('error'); % remove layer
netbasemodel.removeLayer('loss'); % remove layer
netbasemodel.removeLayer('softmax'); % remove layer
netbasemodel.removeLayer('fc5'); % remove layer
% netbasemodel.removeLayer('pool_final'); % remove layer

netbasemodel.meta.normalization.averageImage = meanVal; % imagenet mean values

% netbasemodel.params(1).value = mean(netbasemodel.params(1).value, 3);
% netbasemodel.layers(1).block.size = [7 7 1 64];
% netbasemodel.layers(1).block.stride = [1 1];
% netbasemodel.layers(4).block.stride = [2 2];

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

% for i = 1:numel(netbasemodel.params)
%     fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);
% end
% RFinfo = netbasemodel.getVarReceptiveFields('data');
%% last layer for classification
sName = 'res5_3_relu';
lName = 'conv6';
inputDim = 256;
outputDim = 64;
block = dagnn.Conv('size', [3 3 inputDim outputDim], 'hasBias', false, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f']});
ind = netbasemodel.getParamIndex([lName '_f']);
weights = randn(3, 3, inputDim, outputDim, 'single')*sqrt(2/outputDim);
netbasemodel.params(ind).value = weights;
netbasemodel.params(ind).learningRate = 1;
sName = lName;

lName = 'conv7';
inputDim = outputDim;
outputDim = 3;
block = dagnn.Conv('size', [1 1 inputDim outputDim], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
ind = netbasemodel.getParamIndex([lName '_f']);
weights = randn(1, 1, inputDim, outputDim, 'single')*sqrt(2/outputDim);
netbasemodel.params(ind).value = weights;
netbasemodel.params(ind).learningRate = 1;
ind = netbasemodel.getParamIndex([lName '_b']);
netbasemodel.params(ind).value = zeros([1 outputDim], 'single');
netbasemodel.params(ind).learningRate = 2;
sName = lName;

lossCellList = {'loss', 1};
obj_name = sprintf('loss');
gt_name =  sprintf('label');
netbasemodel.addLayer(obj_name, ...
    DimEmotionLoss('loss', 'reg_L1'), ... softmaxlog logistic
    {sName, gt_name}, obj_name);

% gt_name =  'label';
% obj_name = 'obj_regression_L2';
% netbasemodel.addLayer(obj_name, ...
%     DimEmotionLoss('loss', 'reg_L2'), ...
%     {sName, gt_name}, obj_name);
% 
% obj_name = 'obj_regression_L1';
% netbasemodel.addLayer(obj_name, ...
%     DimEmotionLoss('loss', 'reg_L1'), ...
%     {sName, gt_name}, obj_name);

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
batchSize = 350; % each batch contains one image, around 640x768-pixel resolution
mopts.classifyType='L1'; 

% some parameters should be tuned
opts.batchSize = batchSize;
opts.learningRate = netbasemodel.meta.trainOpts.learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize = opts.batchSize;
%% setup to train network
opts.expDir = fullfile('./exp', 'main002_basemodel');
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
prefixStr = [dataset, '_', netbasemodelName, '_', mopts.classifyType, '_'];

fn = getBatchWrapper_DimEmotion(bopts) ;

rng('default');
opts.gpus = 1;
opts.checkpointFn = [];
% opts.backPropAboveLayerName = 'res6_conv';
opts.backPropAboveLayerName = 'conv1_1';

[netbasemodel, info] = cnntrainDag_DimEmotion_ResNet(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', lossCellList, opts);


