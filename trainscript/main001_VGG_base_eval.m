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
saveFolder = 'main010_basemodel_v2_train';
modelName = 'DimEmotion_resnet_L1_net-epoch-497.mat'; % 497


netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

netbasemodel.layers(38).block = rmfield(netbasemodel.layers(38).block,'ignoreAverage');
netbasemodel.layers(38).block = rmfield(netbasemodel.layers(38).block,'normalise');
netbasemodel.layers(39).block = rmfield(netbasemodel.layers(39).block,'ignoreAverage');
netbasemodel.layers(39).block = rmfield(netbasemodel.layers(39).block,'normalise');

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
./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-245.mat
RMSE: valence:1.01124, arousal:0.75469 dominance:0.75133.
correlation: 0.87359, 0.83544 0.80074.
accuracy:0.77 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-265.mat
RMSE: valence:1.01681, arousal:0.74530 dominance:0.75502.
correlation: 0.87536, 0.83798 0.80307.
accuracy:0.77 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-272.mat
RMSE: valence:0.99825, arousal:0.74704 dominance:0.74173.
correlation: 0.87712, 0.83876 0.80517.
accuracy:0.77 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-320.mat
RMSE: valence:0.98632, arousal:0.73600 dominance:0.74102.
correlation: 0.88115, 0.84309 0.81052.
accuracy:0.78 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-399.mat
RMSE: valence:0.98173, arousal:0.73074 dominance:0.73877.
correlation: 0.88429, 0.84632 0.81482.
accuracy:0.78 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-421.mat
RMSE: valence:0.96283, arousal:0.72474 dominance:0.72200.
correlation: 0.88574, 0.84749 0.81614.
accuracy:0.79 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-422.mat
RMSE: valence:0.96271, arousal:0.72580 dominance:0.72063.
correlation: 0.88599, 0.84782 0.81648.
accuracy:0.78 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-438.mat
RMSE: valence:0.96283, arousal:0.72415 dominance:0.72131.
correlation: 0.88588, 0.84781 0.81640.
accuracy:0.79 (T=0.95)

RMSE: valence:0.96185, arousal:0.72394 dominance:0.72211.
correlation: 0.88620, 0.84800 0.81701.
accuracy:0.79 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-482.mat
RMSE: valence:0.95815, arousal:0.72360 dominance:0.71752.
correlation: 0.88691, 0.84850 0.81776.
accuracy:0.79 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-485.mat
RMSE: valence:0.95852, arousal:0.72597 dominance:0.71733.
correlation: 0.88706, 0.84843 0.81802.
accuracy:0.78 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-492.mat
RMSE: valence:0.95736, arousal:0.72282 dominance:0.71707.
correlation: 0.88712, 0.84869 0.81803.
accuracy:0.78 (T=0.95)

./exp/main010_basemodel_v2_train/DimEmotion_resnet_L1_net-epoch-497.mat
RMSE: valence:0.95684, arousal:0.72218 dominance:0.71722.
correlation: 0.88706, 0.84867 0.81796.
accuracy:0.79 (T=0.95)
%}

