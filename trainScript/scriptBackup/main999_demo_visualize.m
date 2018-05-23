%% add path and setup configuration
clc; clear; close all;

addpath('exportFig');
addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

gpuId=1;
gpuDevice(gpuId);
%% prepare data
mean_value = reshape([129.1863, 104.7624, 93.5940],[1,1,3]); %face
%% load model
saveFolder = 'main013_vggface2_bilinear_v2_ftV1_fc1';
modelName = 'DimEmotion_resnet_L1_net-epoch-444.mat'; % 64

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
%% read image and get result
folder = 'demo_images';
imOrg = imread(fullfile(folder, 'Zuckerburg.jpeg')); % Yao Hussey Shu Jordan Charless Feng Harden nobody Zuckerburg

% im = imresize(imOrg,[224,224]);
% if size(im,3)==1
%     im = repmat(im, [1,1,3,1]);
% end
% im = single(im)-mean_value;
% 
% % feed the original image
% inputs = {'data', gpuArray(single(im))};
% netbasemodel.eval(inputs) ;
% predLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
% predLabel = squeeze(predLabel);
% 
% %feed the flipped image
% inputs = {'data', gpuArray(single(fliplr(im)))};
% netbasemodel.eval(inputs) ;
% tmpLabel_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
% tmpLabelfliplr = squeeze(tmpLabel_fliplr);
% predLabel = 0.5*(predLabel+tmpLabelfliplr);
% predLabel = (predLabel-5)/4;
% figure(1);
% imshow(imOrg);
% title(sprintf('V:%.3f, A:%.3f, D:%.3f',predLabel(1),predLabel(2),predLabel(3)));


folder = 'demo_images';
imList = dir(folder);
for idx = 3:length(imList)
    imOrg = imread(fullfile(folder, imList(idx).name));
    
    %im = single(im);    
    im = imresize(imOrg,[224,224]);
    %im = repmat(im, [1,1,3,1]);   
    im = single(im)-mean_value;    
    
    % feed the original image
    inputs = {'data', gpuArray(single(im))};
    netbasemodel.eval(inputs) ;
    predLabel = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    predLabel = squeeze(predLabel);
    
    %feed the flipped image
    inputs = {'data', gpuArray(single(fliplr(im)))};
    netbasemodel.eval(inputs) ;
    tmpLabel_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
    tmpLabelfliplr = squeeze(tmpLabel_fliplr);    
    predLabel = 0.5*(predLabel+tmpLabelfliplr);
    predLabel = (predLabel-5)/4;
    figure(idx);
    imshow(imOrg);
    title(sprintf('V:%.3f, A:%.3f, D:%.3f',predLabel(1),predLabel(2),predLabel(3)));
    export_fig(sprintf('result_%s',imList(idx).name), '-transparent');
end

%% leaving blank
