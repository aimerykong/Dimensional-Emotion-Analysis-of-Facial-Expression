%% add path and setup configuration
clc; clear; close all;

addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';
gpuId = 4;
gpuDevice(gpuId);
%% prepare data
load(path_to_imdb) ;
meanVal = imdb.meta.mean_value;
imdb.meta.imagesize = [48,48,1];
%% load model
% saveFolder = 'main006_ftMain005_sigmoid';
% modelName = 'DimEmotion_resnet_L1_net-epoch-152.mat';
% saveFolder = 'main007_resnet50_v1';
% modelName = 'DimEmotion_resnet_L1_net-epoch-906.mat';
saveFolder = 'main007_resnet50_v2_ftV1';
modelName = 'DimEmotion_resnet_L1_net-epoch-452.mat';

netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;

netbasemodel.removeLayer('loss_L2'); % remove layer
netbasemodel.removeLayer('loss_L1'); % remove layer
layerTop = 'sigmoidOutput';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;

% netbasemodel.mode = 'test' ; % choices {test, normal}
netbasemodel.mode = 'normal' ;
netbasemodel.conserveMemory = 1;
%% evaluation
fprintf('evaluation\n');
numVal = size(imdb.val.annot,1);
flags_val = zeros(numVal, 1);
predLabel_val = zeros(numVal, 3);
grndLabel_val = imdb.val.annot;
batchSize = 100;
for i = 1:batchSize:numVal
    ed = i+batchSize-1;
    if ed > numVal
        ed = numVal;
    end
    flags_val(i:end) = 1;
    imBatch = single(imdb.val.image(:,:,:,i:ed));
    imBatch = repmat(imBatch, [1,1,3,1]);
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
    imBatch = single(imdb.test.image(:,:,:,i:ed)); 
    imBatch = repmat(imBatch, [1,1,3,1]);   
    imBatch = single(imBatch)-imdb.meta.mean_value;    
    inputs = {'data', gpuArray(single(imBatch))};
    netbasemodel.eval(inputs) ;
    tmpLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    tmpLabel = squeeze(tmpLabel);
    predLabel_test(i:ed,:) = tmpLabel';
end
save('result.mat', 'predLabel_val','grndLabel_val','predLabel_test','grndLabel_test');
%% measure performance
predLabel_val = predLabel_val*9;
grndLabel_val = grndLabel_val*9;
predLabel_test = predLabel_test*9;
grndLabel_test = grndLabel_test*9;

% validation set
%{
predErr_val = predLabel_val-grndLabel_val;
RMSE = sqrt(mean(predErr_val.^2,1));


predErr_val = sqrt(predErr_val(:,1).^2/3 + predErr_val(:,2).^2/3 + predErr_val(:,3).^2/3);
thr = 1;
numCorrect = sum(abs(predErr_val)<thr);
accuracy = numCorrect/numVal;

squares = predErr_val.^2;
rmse = sqrt(mean(squares));
residuals = predLabel_val - grndLabel_val;

figure;
boxplot(residuals, 'Labels', {'Valence','Arousal','Dominance'});
ylabel('Predicted Error');
title('Residuals');
% correlation
fprintf('correlation: %.5f, %.5f %.5f.\n', ...
    corr(predLabel_val(:,1), grndLabel_val(:,1)),...
    corr(predLabel_val(:,2), grndLabel_val(:,2)),...
    corr(predLabel_val(:,3), grndLabel_val(:,3)));
%}
% test set
predErr_test = predLabel_test-grndLabel_test;
RMSE = sqrt(mean(predErr_test.^2,1));
fprintf('RMSE:');
disp(RMSE);

predErr_test = sqrt(predErr_test(:,1).^2/3 + predErr_test(:,2).^2/3 + predErr_test(:,3).^2/3);
thr = 1;
numCorrect = sum(abs(predErr_test)<thr);
accuracy = numCorrect/numVal;

squares = predErr_test.^2;
rmse = sqrt(mean(squares));
residuals = predLabel_test-grndLabel_test;

figure;
boxplot(residuals, 'Labels', {'Valence','Arousal','Dominance'});
ylabel('Predicted Error');
title('Residuals');
% correlation
fprintf('correlation: %.5f, %.5f %.5f.\n', ...
    corr(predLabel_test(:,1), grndLabel_test(:,1)),...
    corr(predLabel_test(:,2), grndLabel_test(:,2)),...
    corr(predLabel_test(:,3), grndLabel_test(:,3)));

% sqrt(mean((predLabel_val - grndLabel_val).^2,1))
% sqrt(mean((predLabel_test - grndLabel_test).^2,1))
%%
% A = imdb.test.annot*9;
% sum(sum(testEmotion-A))



