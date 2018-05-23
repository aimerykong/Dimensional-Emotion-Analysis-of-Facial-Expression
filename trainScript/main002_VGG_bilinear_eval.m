%% add path and setup configuration
clc; clear; close all;

addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';
gpuId = 3;
gpuDevice(gpuId);
%% prepare data
load(path_to_imdb) ;
imdb.meta.mean_value = reshape([129.1863, 104.7624, 93.5940],[1,1,3]); %face
meanVal = imdb.meta.mean_value;
imdb.meta.imagesize = [224,224,3];
imdb.train.annot = imdb.train.annot*9;
imdb.val.annot = imdb.val.annot*9;
imdb.test.annot = imdb.test.annot*9;
%% load model
saveFolder = 'main011_bilinearPool_v1';
modelName = 'DimEmotion_resnet_L1_net-epoch-192.mat'; % 493
saveFolder = 'main011_bilinearPool_v3_ftV1_fc1';
modelName = 'DimEmotion_resnet_L1_net-epoch-500.mat'; % 500


netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

% netbasemodel.layers(38).block = rmfield(netbasemodel.layers(38).block,'ignoreAverage');
% netbasemodel.layers(38).block = rmfield(netbasemodel.layers(38).block,'normalise');
% netbasemodel.layers(39).block = rmfield(netbasemodel.layers(39).block,'ignoreAverage');
% netbasemodel.layers(39).block = rmfield(netbasemodel.layers(39).block,'normalise');

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;

netbasemodel.removeLayer('loss_L2'); % remove layer
netbasemodel.removeLayer('loss_L1'); % remove layer
layerTop = 'output';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;

netbasemodel.mode = 'test' ; % choices {test, normal}
% netbasemodel.mode = 'normal' ;
netbasemodel.conserveMemory = 1;
%% evaluation
numVal = size(imdb.val.annot,1);
flags_val = zeros(numVal, 1);
predLabel_val = zeros(numVal, 3);
grndLabel_val = imdb.val.annot;
batchSize = 200;
% for i = 1:batchSize:numVal
%     ed = i+batchSize-1;
%     if ed > numVal
%         ed = numVal;
%     end
%     flags_val(i:end) = 1;
%     imBatch = single(imdb.val.image(:,:,:,i:ed));
%     imBatch = imresize(imBatch,[224,224]);
%     imBatch = repmat(imBatch, [1,1,3,1]);
%     imBatch = single(imBatch)-imdb.meta.mean_value;    
%     inputs = {'data', gpuArray(single(imBatch))};
%     netbasemodel.eval(inputs) ;
%     tmpLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
%     tmpLabel = squeeze(tmpLabel);
%     predLabel_val(i:ed,:) = tmpLabel';
% end

fprintf('evaluation\n');
numTest = size(imdb.test.annot,1);
predLabel_test = zeros(numTest, 3);
flags_test = zeros(numTest, 1);
grndLabel_test = imdb.test.annot;
for i = 1:batchSize:numTest
    ed = i+batchSize-1;
    if ed > numTest
        ed = numTest;
    end
    flags_test(i:end) = 1;
    imBatch = single(imdb.test.image(:,:,:,i:ed)); 
    imBatch = imresize(imBatch,[224,224]);
    imBatch = repmat(imBatch, [1,1,3,1]);   
    imBatch = single(imBatch)-imdb.meta.mean_value;    
    
    % feed the original image
    inputs = {'data', gpuArray(single(imBatch))};
    netbasemodel.eval(inputs) ;
    tmpLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    tmpLabel = squeeze(tmpLabel);
    
    %feed the flipped image
    inputs = {'data', gpuArray(single(fliplr(imBatch)))};
    netbasemodel.eval(inputs) ;
    tmpLabel_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    tmpLabelfliplr = squeeze(tmpLabel_fliplr);    
    tmpLabel = 0.5*(tmpLabel+tmpLabelfliplr);
    
    predLabel_test(i:ed,:) = tmpLabel';
end
save('result.mat', 'predLabel_val','grndLabel_val','predLabel_test','grndLabel_test');
%% measure performance
fprintf('%s\n',fullfile('./exp', saveFolder, modelName));
% test set
residuals = predLabel_test-grndLabel_test;
RMSE = sqrt(mean(residuals.^2,1));
fprintf('RMSE: valence:%.5f, arousal:%.5f dominance:%.5f.\n', RMSE(1),RMSE(2),RMSE(3));

predErr_test = sqrt(residuals(:,1).^2/3 + residuals(:,2).^2/3 + residuals(:,3).^2/3);


squares = predErr_test.^2;
rmse = sqrt(mean(squares));
residuals = predLabel_test-grndLabel_test;

imgFig = figure(1);
set(imgFig, 'Position',[100,100,900,400]); % [1 1 width height]
subplot(1,2,1);
boxplot(residuals, 'Labels', {'Valence','Arousal','Dominance'});
ylabel('Predicted Error');
title('Residuals');
% correlation
fprintf('correlation: %.5f, %.5f %.5f.\n', ...
    corr(predLabel_test(:,1), grndLabel_test(:,1)),...
    corr(predLabel_test(:,2), grndLabel_test(:,2)),...
    corr(predLabel_test(:,3), grndLabel_test(:,3)));

subplot(1,2,2);
thrList = 0:0.1:2;
accList = [];
for thr = thrList
    numCorrect = sum(abs(predErr_test)<thr);
    accuracy = numCorrect/numTest;
    accList(end+1) = accuracy;
end
plot(thrList,accList);
thr = 0.95;
numCorrect = sum(abs(predErr_test)<thr);
accuracy = numCorrect/numTest;
fprintf('accuracy:%.4f (T=%.2f)\n', accuracy, thr);
title(sprintf('accuracy:%.4f (T=%.2f)\n', accuracy, thr));

save(['result_' strrep(saveFolder,'/',''), strrep(modelName,'/',''), '.mat'],...
    'thrList', 'accList', 'predLabel_test','grndLabel_test', 'numTest');

figname = ['curve' strrep(saveFolder,'/',''), strrep(modelName,'/',''), '.png'];
export_fig(figname);
%% leaving blank
%{
./exp/main011_bilinearPool_v3_ftV1_fc1/DimEmotion_resnet_L1_net-epoch-260.mat
RMSE: valence:0.90184, arousal:0.68290 dominance:0.68597.
correlation: 0.90043, 0.86663 0.83671.
accuracy:0.83 (T=0.95)

RMSE: valence:0.90167, arousal:0.68034 dominance:0.68982.
correlation: 0.90116, 0.86896 0.83846.
accuracy:0.83 (T=0.95)

./exp/main011_bilinearPool_v3_ftV1_fc1/DimEmotion_resnet_L1_net-epoch-472.mat
RMSE: valence:0.90223, arousal:0.67147 dominance:0.68251.
correlation: 0.90029, 0.87068 0.83748.
accuracy:0.83 (T=0.95)

./exp/main011_bilinearPool_v3_ftV1_fc1/DimEmotion_resnet_L1_net-epoch-500.mat
RMSE: valence:0.90108, arousal:0.67111 dominance:0.68155.
correlation: 0.90054, 0.87071 0.83787.
accuracy:0.83 (T=0.95)
%}

