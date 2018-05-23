clear; clc;
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
%% load model
saveFolder = 'main004_ftMain003_sigmoid';
modelName = 'DimEmotion_resnet_L1_net-epoch-984.mat';

netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;

netbasemodel.removeLayer('loss_L2'); % remove layer
netbasemodel.removeLayer('loss_L1'); % remove layer
layerTop = 'sigmoidOutput';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;

netbasemodel.mode = 'test' ; % choices {test, normal}
%netbasemodel.mode = 'normal' ;
%net.conserveMemory = 0;
%% evaluation
fprintf('evaluation\n');
numVal = size(imdb.val.annot,1);
flags_val = zeros(numVal, 1);
predLabel_val = zeros(numVal, 3);
grndLabel_val = imdb.val.annot;
for i = 1:300:numVal
    ed = i+300-1;
    if ed > numVal
        ed = numVal;
    end
    flags_val(i:end) = 1;
    imBatch = single(imdb.val.image(:,:,:,i:ed))-imdb.meta.mean_value;    
    imBatch = single(imBatch)-imdb.meta.mean_value;    
    inputs = {'data', gpuArray(single(imBatch))};
    netbasemodel.eval(inputs) ;
    tmpLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    tmpLabel = squeeze(tmpLabel);
    predLabel_val(i:ed,:) = tmpLabel';
end

fprintf('evaluation\n');
numTest = size(imdb.test.annot,1);
predLabel_test = zeros(numTest, 3);
flags_test = zeros(numTest, 1);
grndLabel_test = imdb.test.annot;
for i = 1:300:numTest
    ed = i+300-1;
    if ed > numTest
        ed = numTest;
    end
    flags_test(i:end) = 1;
    imBatch = single(imdb.test.image(:,:,:,i:ed))-imdb.meta.mean_value;    
    imBatch = single(imBatch)-imdb.meta.mean_value;    
    inputs = {'data', gpuArray(single(imBatch))};
    netbasemodel.eval(inputs) ;
    tmpLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    tmpLabel = squeeze(tmpLabel);
    predLabel_test(i:ed,:) = tmpLabel';
end
save('result.mat', 'predLabel_val','grndLabel_val','predLabel_test','grndLabel_test');
%% measure performance
predLabel_val(predLabel_val<0.1) = 0.1;
predLabel_test(predLabel_test<0.1) = 0.1;

predLabel_val = predLabel_val*10-1;
grndLabel_val = grndLabel_val*9;
predLabel_test = predLabel_test*10-1;
grndLabel_test = grndLabel_test*9;

sqrt(mean((predLabel_val - grndLabel_val).^2,1))
sqrt(mean((predLabel_test - grndLabel_test).^2,1))




%%
% A = imdb.test.annot*9;
% sum(sum(testEmotion-A))



