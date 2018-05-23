%% add path and setup configuration
clc; clear; close all;

addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';
gpuId = 1;
gpuDevice(gpuId);
%% prepare data
load(path_to_imdb) ;
% imdb.meta.mean_value = reshape([129.1863, 104.7624, 93.5940],[1,1,3]); %face
imdb.meta.mean_value = reshape([131.0912 103.8827 91.4953],[1,1,3]); %vggface2
meanVal = imdb.meta.mean_value;
imdb.meta.imagesize = [224,224,3];
imdb.train.annot = imdb.train.annot*9;
imdb.val.annot = imdb.val.annot*9;
imdb.test.annot = imdb.test.annot*9;
%% load model
saveFolder = 'main012_vggface2_v1_basemodel/';
modelName = 'DimEmotion_resnet_L1_net-epoch-492.mat'; % 492

saveFolder = 'main012_vggface2_v3_ftv1_1topConv/';
modelName = 'DimEmotion_resnet_L1_net-epoch-293.mat'; % 61

% saveFolder = 'main012_vggface2_v2_basemodel_2topConv/';
% modelName = 'DimEmotion_resnet_L1_net-epoch-447.mat'; % 390


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
%     imBatch = imresize(imBatch,[224,224]);
%     imBatch = repmat(imBatch, [1,1,3,1]);   
%     imBatch = single(imBatch)-imdb.meta.mean_value;    
    
    
    % feed the original image
    curLabel = 0;
    count = 0;
    for xstart=1:2:5
        for xend=1:2:5
            for ystart=1:2:5
                for yend=1:2:5       
                    imBatchTMP = single(imBatch(ystart:end-yend+1,xstart:end-xend+1,:,:));
                    imBatchTMP = imresize(imBatchTMP,[224,224]);
                    imBatchTMP = repmat(imBatchTMP, [1,1,3,1]);
                    imBatchTMP = single(imBatchTMP)-imdb.meta.mean_value;
                    
                    inputs = {'data', gpuArray(imBatchTMP)};
                    netbasemodel.eval(inputs) ;
                    tmpLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
                    tmpLabel = squeeze(tmpLabel);
                    
                    %feed the flipped image
                    inputs = {'data', gpuArray(single(fliplr(imBatchTMP)))};
                    netbasemodel.eval(inputs) ;
                    tmpLabel_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
                    tmpLabelfliplr = squeeze(tmpLabel_fliplr);
                    tmpLabel = tmpLabel+tmpLabelfliplr;
                    
                    curLabel = curLabel + tmpLabel;
                    count = count + 2;
                end
            end
        end
    end
    curLabel = curLabel / count;
    predLabel_test(i:ed,:) = curLabel';
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
fprintf('accuracy:%.2f (T=%.2f)\n', accuracy, thr);
title(sprintf('accuracy:%.2f (T=%.2f)\n', accuracy, thr));
%% leaving blank
%{
./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-293.mat(new mean)
RMSE: valence:0.87918, arousal:0.69372 dominance:0.68522.
correlation: 0.90650, 0.86220 0.83971.
accuracy:0.83 (T=0.95)

./exp/main012_vggface2_v3_ftv1_1topConv/DimEmotion_resnet_L1_net-epoch-293.mat
RMSE: valence:0.88782, arousal:0.70204 dominance:0.68269.
correlation: 0.90472, 0.85950 0.83939.
accuracy:0.82 (T=0.95)
%}

