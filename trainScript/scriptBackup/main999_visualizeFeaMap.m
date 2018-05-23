%% add path and setup configuration
clc; clear; close all;

addpath('exportFig');
addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

path_to_imdb = 'imdb_DimEmotion.mat';
dataset = 'DimEmotion';
gpuId = 2;
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


% netbasemodel.layers(79).block.stride = [1 1]; % conv4_1_1x1_reduce
% netbasemodel.layers(87).block.stride = [1 1]; % conv4_1_1x1_proj
% dilationRate = 2;
% for idx = 79:140
%     if strcmpi( class(netbasemodel.layers(idx).block), 'dagnn.Conv' ) && netbasemodel.layers(idx).block.size(1) == 3
%         netbasemodel.layers(idx).block.dilate = [dilationRate dilationRate];
%         netbasemodel.layers(idx).block.pad = [dilationRate dilationRate];
%         disp(netbasemodel.layers(idx).name);
%     end
% end
%
% % ResBlock5
% netbasemodel.layers(141).block.stride = [1 1]; % conv5_1_1x1_reduce
% netbasemodel.layers(149).block.stride = [1 1]; % conv5_1_1x1_proj
% dilationRate = 4;
% for idx = 140:168
%     if strcmpi( class(netbasemodel.layers(idx).block), 'dagnn.Conv' ) && netbasemodel.layers(idx).block.size(1) == 3
%         netbasemodel.layers(idx).block.dilate = [dilationRate dilationRate];
%         netbasemodel.layers(idx).block.pad = [dilationRate dilationRate];
%         disp(netbasemodel.layers(idx).name);
%     end
% end


netbasemodel.removeLayer('loss_L2'); % remove layer
netbasemodel.removeLayer('loss_L1'); % remove layer
layerTop = 'output';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;

feamapLayerName = 'conv5_3_3x3_relu';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(feamapLayerName)).outputIndexes).precious = 1;

netbasemodel.mode = 'test' ; % choices {test, normal}
% netbasemodel.mode = 'normal' ;
netbasemodel.conserveMemory = 1;

save2folder = 'testResult';
if ~isdir(save2folder)
    mkdir(save2folder);
end
%% evaluation

for idx = 1:100:length(grndLabel_test)    
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
    
    figure(1);
    hFig=gcf;
    subplot(1,2,1);
    imagesc(imOrg);
    axis off image;
    colormap(gca,'gray');
    title(sprintf('grndV:%.2f,grndA:%.2f,grndD:%.2f \npredV:%.2f,predA:%.2f,predD:%.2f', ...
        grndLabel_test(idx,1), grndLabel_test(idx,2), grndLabel_test(idx,3),...
        predLabel(1), predLabel(2), predLabel(3)));
    
    % hcb1=colorbar;
    maxMap = max(FeaMap,[],3);
    % avgMap = mean(FeaMap,3);
    subplot(1,2,2);
    imagesc(maxMap);
    colormap(gca,'jet')
    %colormap(jet);
    axis off image;
    
    export_fig(fullfile(save2folder, sprintf('result_%04d',idx)), '-transparent');
end
%% leaving blank
%{
%}

