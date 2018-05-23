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
gpuId = 2;
gpuDevice(gpuId);
%% prepare data
load(path_to_imdb) ;
imdb.meta.mean_value = reshape([129.1863, 104.7624, 93.5940],[1,1,3]); %face
% imdb.meta.mean_value = reshape([131.0912 103.8827 91.4953],[1,1,3]); %vggface2
meanVal = imdb.meta.mean_value;
imdb.meta.imagesize = [224,224,3];
imdb.train.annot = imdb.train.annot*9;
imdb.val.annot = imdb.val.annot*9;
imdb.test.annot = imdb.test.annot*9;
%% load model
saveFolder = 'main012_vggface2_v1_basemodel/';
modelName = 'DimEmotion_resnet_L1_net-epoch-492.mat'; % 492

saveFolder = 'main012_vggface2_v3_ftv1_1topConv/';
modelName = 'DimEmotion_resnet_L1_net-epoch-500.mat'; % 333


netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;


netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;

netbasemodel.removeLayer('loss_L2'); % remove layer
netbasemodel.removeLayer('loss_L1'); % remove layer

layerTop = 'block6_output';
if ~isnan(netbasemodel.getLayerIndex(layerTop))
    layerTop = 'block6_output';
    netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

rmLayerName = 'block7_conv';
if ~isnan(netbasemodel.getLayerIndex(rmLayerName))
    layerTop = 'block7_conv';
    netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end


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
fprintf('correlation: %.5f, %.5f, %.5f\n', ...
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
./exp/main012_vggface2_v1_basemodel/DimEmotion_resnet_L1_net-epoch-390.mat
RMSE: valence:0.89803, arousal:0.70995 dominance:0.69553.
correlation: 0.90164, 0.85456 0.83305.
accuracy:0.81 (T=0.95)

./exp/main012_vggface2_v1_basemodel/DimEmotion_resnet_L1_net-epoch-441.mat
RMSE: valence:0.89454, arousal:0.70643 dominance:0.69013.
correlation: 0.90192, 0.85558 0.83402.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v1_basemodel/DimEmotion_resnet_L1_net-epoch-447.mat
RMSE: valence:0.89398, arousal:0.70720 dominance:0.68759.
correlation: 0.90220, 0.85549 0.83456.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v1_basemodel/DimEmotion_resnet_L1_net-epoch-455.mat
RMSE: valence:0.89201, arousal:0.70574 dominance:0.68881.
correlation: 0.90269, 0.85618 0.83479.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v1_basemodel/DimEmotion_resnet_L1_net-epoch-488.mat
RMSE: valence:0.89270, arousal:0.70499 dominance:0.68893.
correlation: 0.90253, 0.85640 0.83422.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v1_basemodel/DimEmotion_resnet_L1_net-epoch-489.mat
RMSE: valence:0.89169, arousal:0.70570 dominance:0.68603.
correlation: 0.90264, 0.85595 0.83500.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v1_basemodel/DimEmotion_resnet_L1_net-epoch-492.mat
RMSE: valence:0.88914, arousal:0.70570 dominance:0.68703.
correlation: 0.90321, 0.85617 0.83540.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v1_basemodel/DimEmotion_resnet_L1_net-epoch-500.mat
RMSE: valence:0.88998, arousal:0.70759 dominance:0.68573.
correlation: 0.90298, 0.85549 0.83521.
accuracy:0.82 (T=0.95)








./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-7.mat
RMSE: valence:0.89934, arousal:0.70778 dominance:0.69561.
correlation: 0.90306, 0.85704 0.83547.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-25.mat
RMSE: valence:0.89064, arousal:0.70347 dominance:0.68937.
correlation: 0.90297, 0.85703 0.83512.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-38.mat
RMSE: valence:0.88892, arousal:0.70450 dominance:0.68619.
correlation: 0.90331, 0.85721 0.83495.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-54.mat
RMSE: valence:0.89458, arousal:0.70607 dominance:0.69882.
correlation: 0.90258, 0.85730 0.83445.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-60.mat
RMSE: valence:0.89271, arousal:0.70315 dominance:0.69001.
correlation: 0.90289, 0.85706 0.83507.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-61.mat
RMSE: valence:0.88799, arousal:0.70345 dominance:0.69075.
correlation: 0.90355, 0.85769 0.83526.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-62.mat
RMSE: valence:0.88774, arousal:0.70303 dominance:0.68755.
correlation: 0.90350, 0.85726 0.83526.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-77.mat
RMSE: valence:0.90313, arousal:0.70541 dominance:0.69626.
correlation: 0.90241, 0.85688 0.83478.
accuracy:0.81 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-78.mat
RMSE: valence:0.88723, arousal:0.70252 dominance:0.68480.
correlation: 0.90382, 0.85746 0.83672.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-87.mat
RMSE: valence:0.88808, arousal:0.70240 dominance:0.68577.
correlation: 0.90345, 0.85735 0.83563.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-88.mat
RMSE: valence:0.89219, arousal:0.70658 dominance:0.68514.
correlation: 0.90349, 0.85735 0.83620.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-168.mat
RMSE: valence:0.89371, arousal:0.70179 dominance:0.69576.
correlation: 0.90314, 0.85841 0.83505.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-197.mat
RMSE: valence:0.89185, arousal:0.69989 dominance:0.68666.
correlation: 0.90278, 0.85865 0.83558.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-208.mat
RMSE: valence:0.88950, arousal:0.69940 dominance:0.68599.
correlation: 0.90317, 0.85863 0.83582.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-263.mat
RMSE: valence:0.87530, arousal:0.69235 dominance:0.68350.
correlation: 0.90672, 0.86220 0.83984.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-263.mat(new mean)
RMSE: valence:0.87706, arousal:0.69310 dominance:0.68599.
correlation: 0.90662, 0.86208 0.83947.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-269.mat
RMSE: valence:0.87799, arousal:0.69245 dominance:0.68526.
correlation: 0.90627, 0.86222 0.83936.
accuracy:0.83 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-293.mat
RMSE: valence:0.87671, arousal:0.69275 dominance:0.68273.
correlation: 0.90674, 0.86239 0.84019.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-293.mat(new mean)
RMSE: valence:0.87918, arousal:0.69372 dominance:0.68522.
correlation: 0.90650, 0.86220 0.83971.
accuracy:0.83 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-303.mat(new mean)
RMSE: valence:0.87610, arousal:0.69254 dominance:0.68999.
correlation: 0.90792, 0.86307 0.84108.
accuracy:0.82 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-303.mat
RMSE: valence:0.87299, arousal:0.69154 dominance:0.68688.
correlation: 0.90812, 0.86323 0.84151.
accuracy:0.83 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-305.mat
RMSE: valence:0.87183, arousal:0.69025 dominance:0.67979.
correlation: 0.90763, 0.86295 0.84121.
accuracy:0.83 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-305.mat(new mean)
RMSE: valence:0.87369, arousal:0.69090 dominance:0.68209.
correlation: 0.90751, 0.86282 0.84080.
accuracy:0.83 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-333.mat
RMSE: valence:0.86805, arousal:0.68958 dominance:0.67838.
correlation: 0.90841, 0.86416, 0.84205
accuracy:0.8279 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-343.mat
RMSE: valence:0.87419, arousal:0.68847 dominance:0.68061.
correlation: 0.90708, 0.86352 0.83991.
accuracy:0.83 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-439.mat
RMSE: valence:0.86386, arousal:0.68851 dominance:0.67653.
correlation: 0.90916, 0.86440, 0.84289
accuracy:0.8313 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-454.mat
RMSE: valence:0.86718, arousal:0.68685 dominance:0.68171.
correlation: 0.90914, 0.86516, 0.84268
accuracy:0.8302 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-500.mat
RMSE: valence:0.86236, arousal:0.68665 dominance:0.67140.
correlation: 0.90937, 0.86437, 0.84350
accuracy:0.8316 (T=0.95)
%}

