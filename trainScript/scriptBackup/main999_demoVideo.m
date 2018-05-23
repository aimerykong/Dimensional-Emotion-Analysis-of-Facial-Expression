%% add path and setup configuration
clc; clear; close all;

addpath('exportFig');
addpath(genpath('/home/skong2/projects/MarkovCNN/libs'));
path_to_matconvnet = '/home/skong2/projects/MarkovCNN/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
%% read data
videoFileName = './demo_video2.mp4';

[~,curname,curext] = fileparts(videoFileName);
if ~exist([curname '_small.avi'], 'file')    
    v = VideoReader(videoFileName);
    
    targetVideo = VideoWriter([curname '_small.avi']);
    open(targetVideo);
    i = 1;
    while hasFrame(v) 
        i = i + 1;
        curframe = readFrame(v);
        curframe = imresize(curframe, 0.25);
        curframe = imrotate(curframe, 90);
        %curframe = curframe(95:318,3:226,:);
        curframe = curframe(95:318,12:235,:);
        writeVideo(targetVideo, curframe);
    end
    whos curframe
    close(targetVideo);
end


if ~exist([curname '_small.mat'], 'file')    
    v = VideoReader([curname '_small.avi']);
    matFileName = [curname '_small.mat'];
    mat = zeros(224,224,3);
    i = 1;
    while hasFrame(v)
        curframe = readFrame(v);
        mat(:,:,:,i) = curframe;
        i = i + 1;
    end
    save(matFileName, 'mat');
end

refPieChart = imread('ref.jpg');
load([curname '_small.mat']);
meanVal = reshape([129.1863, 104.7624, 93.5940],[1,1,3]); %face
%% read model
gpuId = 4;
gpuDevice(gpuId);

saveFolder = 'main013_vggface2_bilinear_v2_ftV1_fc1';
modelName = 'DimEmotion_resnet_L1_net-epoch-500.mat'; % 64

model = load( fullfile('./exp', saveFolder, modelName) );
model = model.net;
model = dagnn.DagNN.loadobj(model);
model.move('gpu') ;

model.removeLayer('loss_L2'); % remove layer
model.removeLayer('loss_L1'); % remove layer
layerTop = 'output';
model.vars(model.layers(model.getLayerIndex(layerTop)).outputIndexes).precious = 1;

feamapLayerName = 'conv5_3_3x3_relu';
model.vars(model.layers(model.getLayerIndex(feamapLayerName)).outputIndexes).precious = 1;

model.mode = 'test' ; % choices {test, normal}
% netbasemodel.mode = 'normal' ;
model.conserveMemory = 1;
%% analysis
mat = single(mat);
mat = imresize(mat,[224,224]);
%mat = repmat(mat, [1,1,3,1]);
mat = single(mat)-meanVal;

inputs = {'data', gpuArray(single(mat))};
model.eval(inputs) ;
predLabel = gather(model.vars(model.layers(model.getLayerIndex(layerTop)).outputIndexes).value);
predLabel = squeeze(predLabel);
FeaMap = gather(model.vars(model.layers(model.getLayerIndex(feamapLayerName)).outputIndexes).value);
FeaMap = squeeze(FeaMap);
%feed the flipped image
inputs = {'data', gpuArray(single(fliplr(mat)))};
model.eval(inputs) ;
tmpLabel_fliplr = gather(model.vars(model.layers(model.getLayerIndex(layerTop)).outputIndexes).value);
FeaMap_fliplr = gather(model.vars(model.layers(model.getLayerIndex(feamapLayerName)).outputIndexes).value);
tmpLabelfliplr = squeeze(tmpLabel_fliplr);
FeaMap_fliplr = squeeze(FeaMap_fliplr);
predLabel = 0.5*(predLabel+tmpLabelfliplr);
FeaMap = 0.5*(FeaMap+FeaMap_fliplr);
predLabel = (predLabel-5)/4;
%% visualization and save
imgFig = figure(1); clf;
set(imgFig, 'Position', [10 10 1200 900]) % [1 1 width height]

% subplot(2,2,2);
% imshow(refPieChart);

frameIdx = 1;
subplot(2,2,1);
imshow(uint8(bsxfun(@plus,mat(:,:,:,frameIdx),meanVal)));
curPrediction = predLabel(:,frameIdx);
title(sprintf('V:%.3f, A:%.3f, D:%.3f',curPrediction(1),curPrediction(2),curPrediction(3)));

demoVideo.frame = uint8(single(mat)+meanVal);
demoVideo.prediction = curPrediction(:);

subplot(2,2,2); 
plot(curPrediction(1),curPrediction(2),'.');
image('CData', refPieChart, 'XData',[-1.1 1.1], 'YData',[1 -1]);
hold on;
grid on;
pbaspect([1 1 1]);
axis([-1.2 1.2 -1.1 1.1]); % [xmin xmax ymin ymax zmin zmax]
xlabel('Valence');
ylabel('Arousal');
title('Two-Dimensional Emotion Model');

subplot(2,1,2); 
plot3(curPrediction(1),curPrediction(2), curPrediction(3), '.');
grid on;
%pbaspect([1 1 1]);
axis([-1 1 -1 1 -1 1]); % [xmin xmax ymin ymax zmin zmax]
xlabel('Valence');
ylabel('Arousal');
zlabel('dominance');
title('Three-Dimensional Emotion Model')


v = VideoWriter([curname '_display.avi']); 
gifMat = [];
gifName = [curname '_display.gif'];
gifNameHighRes = [curname '_displayHighRes.gif'];
open(v);
for frameIdx = 2:size(mat,4)
    subplot(2,2,1);
    imshow(uint8(bsxfun(@plus,mat(:,:,:,frameIdx),meanVal)));
    curPrediction = predLabel(:,frameIdx);
    title(sprintf('V:%.2f, A:%.2f, D:%.2f',curPrediction(1),curPrediction(2),curPrediction(3)));
        
    demoVideo.prediction = [demoVideo.prediction, curPrediction(:)];
    
    ptrA = predLabel(:,frameIdx-1);
    ptrB = predLabel(:,frameIdx);
        
    subplot(2,2,2);    
    line([ptrA(1);ptrB(1)],[ptrA(2);ptrB(2)],'linestyle','-','linewidth',2); % line([x1,x2],[y1,y2]);    
    plot(ptrB(1),ptrB(2), '.', 'MarkerSize', 5, 'MarkerFaceColor', [1,0,0]);
    
    subplot(2,1,2);    
    line([ptrA(1);ptrB(1)],[ptrA(2);ptrB(2)],[ptrA(3);ptrB(3)],'linestyle','-','linewidth',5); % line([x1,x2],[y1,y2]);    
    
    frame = getframe(gcf);   
    writeVideo(v,frame);
    
    imHighRes = frame2im(frame);      
    [imindHighRes,cmHighRes] = rgb2ind(imHighRes,256); 
    
    im = [imHighRes(45:440,200:500,:) imHighRes(45:440,610:1080,:)];
    im = imresize(im,0.5);
    [imind,cm] = rgb2ind(im,256); 
    
    if frameIdx == 2
        imwrite(imind, cm, gifName, 'gif', 'Loopcount', inf);
        imwrite(imindHighRes, cmHighRes, gifNameHighRes, 'gif', 'Loopcount', inf);
    elseif mod(frameIdx,4)==0 && frameIdx<70
        imwrite(imind, cm, gifName, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);
        imwrite(imindHighRes, cmHighRes, gifNameHighRes, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);        
    end
end
close(v);
save([curname '_demoVideoMat.mat'], 'demoVideo');

%% leaving blank