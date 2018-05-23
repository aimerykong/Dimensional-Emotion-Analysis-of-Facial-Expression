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
gpuId=3;
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
saveFolder = 'main013_vggface2_bilinear_v1';
modelName = 'DimEmotion_resnet_L1_net-epoch-500.mat'; % 500

saveFolder = 'main013_vggface2_bilinear_v2_ftV1_fc1';
modelName = 'DimEmotion_resnet_L1_net-epoch-500.mat'; % 444



netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;

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
RMSE: valence:0.85883, arousal:0.69189 dominance:0.68800.
correlation: 0.91032, 0.86186 0.83733.
accuracy:0.82 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-24.mat
RMSE: valence:0.86346, arousal:0.67995 dominance:0.69237.
correlation: 0.91344, 0.87021 0.84370.
accuracy:0.83 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-40.mat
RMSE: valence:0.83584, arousal:0.65479 dominance:0.67777.
correlation: 0.91724, 0.87882 0.84940.
accuracy:0.84 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-43.mat
RMSE: valence:0.81774, arousal:0.65040 dominance:0.65460.
correlation: 0.91899, 0.87922 0.85316.
accuracy:0.85 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-47.mat
RMSE: valence:0.81670, arousal:0.65186 dominance:0.65810.
correlation: 0.91992, 0.88003 0.85449.
accuracy:0.85 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-50.mat
RMSE: valence:0.85529, arousal:0.64699 dominance:0.70092.
correlation: 0.91954, 0.88313 0.85171.
accuracy:0.83 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-60.mat
RMSE: valence:0.80712, arousal:0.63663 dominance:0.64710. 
correlation: 0.92203, 0.88505 0.85918.
accuracy:0.85 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-64.mat
RMSE: valence:0.79613, arousal:0.63728 dominance:0.63812.
correlation: 0.92319, 0.88427 0.86067.
accuracy:0.86 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-99.mat
RMSE: valence:0.78998, arousal:0.61916 dominance:0.63878.
correlation: 0.92595, 0.89201 0.86413.
accuracy:0.87 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-162.mat
RMSE: valence:0.77171, arousal:0.60505 dominance:0.62655.
correlation: 0.92847, 0.89698 0.86844.
accuracy:0.87 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-242.mat
RMSE: valence:0.77438, arousal:0.59462 dominance:0.62797.
correlation: 0.93025, 0.90129 0.87241.
accuracy:0.88 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-346.mat
RMSE: valence:0.75379, arousal:0.58504 dominance:0.61338.
correlation: 0.93202, 0.90439 0.87506.
accuracy:0.88 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-459.mat
RMSE: valence:0.75539, arousal:0.58523 dominance:0.61229.
correlation: 0.93219, 0.90417 0.87609.
accuracy:0.88 (T=0.95)

./exp/main013_vggface2_bilinear_v1/DimEmotion_resnet_L1_net-epoch-500.mat
RMSE: valence:0.75277, arousal:0.58194 dominance:0.60634.
correlation: 0.93242, 0.90511 0.87710.
accuracy:0.88 (T=0.95)





./exp/main013_vggface2_bilinear_v2_ftV1_fc1/DimEmotion_resnet_L1_net-epoch-28.mat
RMSE: valence:0.79012, arousal:0.62476 dominance:0.63991.
correlation: 0.92486, 0.88914 0.86188.
accuracy:0.86 (T=0.95)

./exp/main013_vggface2_bilinear_v2_ftV1_fc1/DimEmotion_resnet_L1_net-epoch-89.mat
RMSE: valence:0.77935, arousal:0.60030 dominance:0.63150.
correlation: 0.92762, 0.89847 0.86862.
accuracy:0.87 (T=0.95)

./exp/main013_vggface2_bilinear_v2_ftV1_fc1/DimEmotion_resnet_L1_net-epoch-267.mat
RMSE: valence:0.75536, arousal:0.58324 dominance:0.60949.
correlation: 0.93214, 0.90505 0.87441.
accuracy:0.88 (T=0.95)

./exp/main013_vggface2_bilinear_v2_ftV1_fc1/DimEmotion_resnet_L1_net-epoch-444.mat
RMSE: valence:0.74420, arousal:0.57892 dominance:0.60678.
correlation: 0.93361, 0.90696 0.87661.
accuracy:0.89 (T=0.95)

./exp/main013_vggface2_bilinear_v2_ftV1_fc1/DimEmotion_resnet_L1_net-epoch-500.mat
RMSE: valence:0.74633, arousal:0.57445 dominance:0.60172.
correlation: 0.93313, 0.90721 0.87701.
accuracy:0.89 (T=0.95)

%}

