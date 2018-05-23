%% add path and setup configuration
clc; clear; close all;

addpath('../libs/exportFig');
addpath('../libs/layerExt');
addpath('../libs/myFunctions');
path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
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
saveFolder = 'main014_2dimModel_vgg16_v1_forward/';
modelName = 'DimEmotion_resnet_L1_net-epoch-100.mat'; % 82

saveFolder = 'main014_2dimModel_vgg16_v2_bilinear/';
modelName = 'DimEmotion_resnet_L1_net-epoch-100.mat'; % 100

saveFolder = 'main015_2dimModel_res50_v1_forward';
modelName = 'DimEmotion_resnet_L1_net-epoch-100.mat'; % 79

saveFolder = 'main015_2dimModel_res50_v2_bilinear';
modelName = 'DimEmotion_resnet_L1_net-epoch-54.mat'; % 54 61 100


netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;

netbasemodel.removeLayer('loss_L2'); % remove layer
netbasemodel.removeLayer('loss_L1'); % remove layer
layerTop = 'output';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;

netbasemodel.mode = 'test' ; % choices {test, normal}
netbasemodel.conserveMemory = 1;
%% evaluation
batchSize = 200;

fprintf('evaluation\n');
numTest = size(imdb.test.annot,1);
predLabel_test = zeros(numTest, 2);
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

%% measure performance
fprintf('%s\n',fullfile('./exp', saveFolder, modelName));
% test set
residuals = predLabel_test-grndLabel_test(:,1:2);
RMSE = sqrt(mean(residuals.^2,1));
fprintf('RMSE: valence:%.5f, arousal:%.5f.\n', RMSE(1),RMSE(2));

predErr_test = sqrt(residuals(:,1).^2/2 + residuals(:,2).^2/2);

squares = predErr_test.^2;
rmse = sqrt(mean(squares));
residuals = predLabel_test-grndLabel_test(:,1:2);

imgFig = figure(1);
set(imgFig, 'Position',[100,100,900,400]); % [1 1 width height]
subplot(1,2,1);
boxplot(residuals, 'Labels', {'Valence','Arousal'});
ylabel('Predicted Error');
title('Residuals');
% correlation
fprintf('correlation: %.5f, %.5f.\n', ...
    corr(predLabel_test(:,1), grndLabel_test(:,1)),...
    corr(predLabel_test(:,2), grndLabel_test(:,2)));

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
./exp/main014_2dimModel_vgg16_v1_forward/DimEmotion_resnet_L1_net-epoch-82.mat
RMSE: valence:0.92887, arousal:0.70218.
correlation: 0.89482, 0.85917.
accuracy:0.79 (T=0.95)

./exp/main014_2dimModel_vgg16_v1_forward/DimEmotion_resnet_L1_net-epoch-100.mat
RMSE: valence:0.92381, arousal:0.69702.
correlation: 0.89536, 0.85973.
accuracy:0.80 (T=0.95)




./exp/main014_2dimModel_vgg16_v2_bilinear/DimEmotion_resnet_L1_net-epoch-61.mat
RMSE: valence:0.88423, arousal:0.65818.
correlation: 0.90704, 0.87644.
accuracy:0.82 (T=0.95)

./exp/main014_2dimModel_vgg16_v2_bilinear/DimEmotion_resnet_L1_net-epoch-100.mat
RMSE: valence:0.87395, arousal:0.65678.
correlation: 0.90713, 0.87674.
accuracy:0.83 (T=0.95)




./exp/main015_2dimModel_res50_v1_forward/DimEmotion_resnet_L1_net-epoch-79.mat
RMSE: valence:0.84115, arousal:0.66559.
correlation: 0.91513, 0.87426.
accuracy:0.83 (T=0.95)

./exp/main015_2dimModel_res50_v1_forward/DimEmotion_resnet_L1_net-epoch-100.mat
RMSE: valence:0.84067, arousal:0.66772.
correlation: 0.91488, 0.87444.
accuracy:0.83 (T=0.95)





./exp/main015_2dimModel_res50_v2_bilinear/DimEmotion_resnet_L1_net-epoch-54.mat
RMSE: valence:0.74111, arousal:0.57488.
correlation: 0.93418, 0.90796.
accuracy:0.88 (T=0.95)

./exp/main015_2dimModel_res50_v2_bilinear/DimEmotion_resnet_L1_net-epoch-61.mat
RMSE: valence:0.75255, arousal:0.57643.
correlation: 0.93319, 0.90785.
accuracy:0.88 (T=0.95)

./exp/main015_2dimModel_res50_v2_bilinear/DimEmotion_resnet_L1_net-epoch-100.mat
RMSE: valence:0.74136, arousal:0.57550.
correlation: 0.93404, 0.90829.
accuracy:0.89 (T=0.95)
%}

