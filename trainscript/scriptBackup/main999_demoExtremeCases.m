%% add path and setup configuration
clc; clear; close all;

addpath('exportFig');
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
grndLabel_test = imdb.test.annot;
%% load model
saveFolder = 'main013_vggface2_bilinear_v2_ftV1_fc1';
modelName = 'DimEmotion_resnet_L1_net-epoch-500.mat'; % 64

netbasemodel = load( fullfile('./exp', saveFolder, modelName) );
netbasemodel = netbasemodel.net;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;


netbasemodel.removeLayer('loss_L2'); % remove layer
netbasemodel.removeLayer('loss_L1'); % remove layer
layerTop = 'output';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;

feamapLayerName = 'conv5_3_3x3_relu';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).precious = 1;

netbasemodel.mode = 'test' ; % choices {test, normal}
% netbasemodel.mode = 'normal' ;
netbasemodel.conserveMemory = 1;
%% get ten best and worst cases for specific dimension
save2folder = 'demoExtremeCase';
if ~isdir(save2folder)
    mkdir(save2folder);
end

curmat = load('result_main013_vggface2_bilinear_v2_ftV1_fc1DimEmotion_resnet_L1_net-epoch-500.mat.mat');
residuals = curmat.predLabel_test-curmat.grndLabel_test;
sumResidual = mean(residuals.^2,2);
RMSE = sqrt(mean(residuals.^2,1));
caseNum = 10;
dimName = {'valence', 'arousal', 'dominance'};
for dimIdx = 1:3
    %% good cases
    [valTMP, idxTMP] = sort(abs(residuals(:,dimIdx)), 'ascend');
    idx = idxTMP(1:10);
    imOrg = single(imdb.test.image(:,:,:,idx));
    im = imresize(imOrg,[224,224]);
    im = repmat(im, [1,1,3,1]);
    im = single(im)-imdb.meta.mean_value;
    
    % feed the original image
    inputs = {'data', gpuArray(single(im))};
    netbasemodel.eval(inputs) ;
    predLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    predLabel = squeeze(predLabel);
    FeaMap = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).value);
    FeaMap = squeeze(FeaMap);
    %feed the flipped image
    inputs = {'data', gpuArray(single(fliplr(im)))};
    netbasemodel.eval(inputs) ;
    tmpLabel_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    FeaMap_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).value);
    tmpLabelfliplr = squeeze(tmpLabel_fliplr);
    FeaMap_fliplr = squeeze(FeaMap_fliplr);
    predLabel = 0.5*(predLabel+tmpLabelfliplr);
    FeaMap = 0.5*(FeaMap+FeaMap_fliplr);
    
    maxMap = max(FeaMap,[],3);
    avgMap = mean(FeaMap,3);
    
    for j = 1:caseNum
        imgFig = figure(1); clf;
        set(imgFig, 'Position', [100 100 700 500]) % [1 1 width height]
        
        subplot(1,3,1);
        imagesc(imOrg(:,:,:,j)); axis off image; colormap(gca,'gray');
        title(sprintf('grnd-V:%.2f,A:%.2f,D:%.2f\npred-V:%.2f,A:%.2f,D:%.2f', ...
            grndLabel_test(idxTMP(j),1), grndLabel_test(idxTMP(j),2), grndLabel_test(idxTMP(j),3),...
            predLabel(1,j), predLabel(2,j), predLabel(3,j)));
        
        subplot(1,3,2);
        imagesc(maxMap(:,:,:,j)); colormap(gca,'jet'); axis off image;
        title('max response');
        
        subplot(1,3,3);
        imagesc(avgMap(:,:,:,j)); colormap(gca,'jet'); axis off image;
        title('average response');
        
        export_fig(fullfile(save2folder, sprintf('%s_good%02d.png', dimName{dimIdx}, j)),...
            '-transparent');
    end
    %% bad cases
    [valTMP, idxTMP] = sort(abs(residuals(:,dimIdx)), 'descend');
    idx = idxTMP(1:10);
    imOrg = single(imdb.test.image(:,:,:,idx));
    im = imresize(imOrg,[224,224]);
    im = repmat(im, [1,1,3,1]);
    im = single(im)-imdb.meta.mean_value;
    
    % feed the original image
    inputs = {'data', gpuArray(single(im))};
    netbasemodel.eval(inputs) ;
    predLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    predLabel = squeeze(predLabel);
    FeaMap = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).value);
    FeaMap = squeeze(FeaMap);
    %feed the flipped image
    inputs = {'data', gpuArray(single(fliplr(im)))};
    netbasemodel.eval(inputs) ;
    tmpLabel_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    FeaMap_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).value);
    tmpLabelfliplr = squeeze(tmpLabel_fliplr);
    FeaMap_fliplr = squeeze(FeaMap_fliplr);
    predLabel = 0.5*(predLabel+tmpLabelfliplr);
    FeaMap = 0.5*(FeaMap+FeaMap_fliplr);
    
    maxMap = max(FeaMap,[],3);
    avgMap = mean(FeaMap,3);
    
    for j = 1:caseNum
        imgFig = figure(1); clf;
        set(imgFig, 'Position', [100 100 700 500]) % [1 1 width height]
        
        subplot(1,3,1);
        imagesc(imOrg(:,:,:,j)); axis off image; colormap(gca,'gray');
        title(sprintf('grnd-V:%.2f,A:%.2f,D:%.2f\npred-V:%.2f,A:%.2f,D:%.2f', ...
            grndLabel_test(idxTMP(j),1), grndLabel_test(idxTMP(j),2), grndLabel_test(idxTMP(j),3),...
            predLabel(1,j), predLabel(2,j), predLabel(3,j)));
        
        subplot(1,3,2);
        imagesc(maxMap(:,:,:,j)); colormap(gca,'jet'); axis off image;
        title('max response');
        
        subplot(1,3,3);
        imagesc(avgMap(:,:,:,j)); colormap(gca,'jet'); axis off image;
        title('average response');
        
        export_fig(fullfile(save2folder, sprintf('%s_bad%02d.png', dimName{dimIdx}, j)),...
            '-transparent');
    end
end
%% get ten best and worst cases for summed dimensions
% good cases
[valTMP, idxTMP] = sort(sumResidual,'ascend');
idx = idxTMP(1:10);
imOrg = single(imdb.test.image(:,:,:,idx));
im = imresize(imOrg,[224,224]);
im = repmat(im, [1,1,3,1]);
im = single(im)-imdb.meta.mean_value;

% feed the original image
inputs = {'data', gpuArray(single(im))};
netbasemodel.eval(inputs) ;
predLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
predLabel = squeeze(predLabel);
FeaMap = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).value);
FeaMap = squeeze(FeaMap);
%feed the flipped image
inputs = {'data', gpuArray(single(fliplr(im)))};
netbasemodel.eval(inputs) ;
tmpLabel_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
FeaMap_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).value);
tmpLabelfliplr = squeeze(tmpLabel_fliplr);
FeaMap_fliplr = squeeze(FeaMap_fliplr);
predLabel = 0.5*(predLabel+tmpLabelfliplr);
FeaMap = 0.5*(FeaMap+FeaMap_fliplr);

maxMap = max(FeaMap,[],3);
avgMap = mean(FeaMap,3);

for j = 1:caseNum
    imgFig = figure(1); clf;
    set(imgFig, 'Position', [100 100 700 500]) % [1 1 width height]
    
    subplot(1,3,1);
    imagesc(imOrg(:,:,:,j)); axis off image; colormap(gca,'gray');
    title(sprintf('grnd-V:%.2f,A:%.2f,D:%.2f\npred-V:%.2f,A:%.2f,D:%.2f', ...
        grndLabel_test(idxTMP(j),1), grndLabel_test(idxTMP(j),2), grndLabel_test(idxTMP(j),3),...
        predLabel(1,j), predLabel(2,j), predLabel(3,j)));
    
    subplot(1,3,2);
    imagesc(maxMap(:,:,:,j)); colormap(gca,'jet'); axis off image;
    title('max response');
    
    subplot(1,3,3);
    imagesc(avgMap(:,:,:,j)); colormap(gca,'jet'); axis off image;
    title('average response');
    
    export_fig(fullfile(save2folder, sprintf('overall_good%02d.png', j)),...
        '-transparent');
end
% bad cases
[valTMP, idxTMP] = sort(sumResidual, 'descend');
idx = idxTMP(1:10);
imOrg = single(imdb.test.image(:,:,:,idx));
im = imresize(imOrg,[224,224]);
im = repmat(im, [1,1,3,1]);
im = single(im)-imdb.meta.mean_value;

% feed the original image
inputs = {'data', gpuArray(single(im))};
netbasemodel.eval(inputs) ;
predLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
predLabel = squeeze(predLabel);
FeaMap = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).value);
FeaMap = squeeze(FeaMap);
%feed the flipped image
inputs = {'data', gpuArray(single(fliplr(im)))};
netbasemodel.eval(inputs) ;
tmpLabel_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
FeaMap_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).value);
tmpLabelfliplr = squeeze(tmpLabel_fliplr);
FeaMap_fliplr = squeeze(FeaMap_fliplr);
predLabel = 0.5*(predLabel+tmpLabelfliplr);
FeaMap = 0.5*(FeaMap+FeaMap_fliplr);

maxMap = max(FeaMap,[],3);
avgMap = mean(FeaMap,3);

for j = 1:caseNum
    imgFig = figure(1); clf;
    set(imgFig, 'Position', [100 100 700 500]) % [1 1 width height]
    
    subplot(1,3,1);
    imagesc(imOrg(:,:,:,j)); axis off image; colormap(gca,'gray');
    title(sprintf('grnd-V:%.2f,A:%.2f,D:%.2f\npred-V:%.2f,A:%.2f,D:%.2f', ...
        grndLabel_test(idxTMP(j),1), grndLabel_test(idxTMP(j),2), grndLabel_test(idxTMP(j),3),...
        predLabel(1,j), predLabel(2,j), predLabel(3,j)));
    
    subplot(1,3,2);
    imagesc(maxMap(:,:,:,j)); colormap(gca,'jet'); axis off image;
    title('max response');
    
    subplot(1,3,3);
    imagesc(avgMap(:,:,:,j)); colormap(gca,'jet'); axis off image;
    title('average response');
    
    export_fig(fullfile(save2folder, sprintf('overall_bad%02d.png', j)),...
        '-transparent');
end

%% leaving blank
%{
%}

