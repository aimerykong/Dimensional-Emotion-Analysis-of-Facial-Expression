clc;
close all;
addpath('exportFig');

if ~exist('demoVideo','var')
    load('demo_video2_demoVideoMat.mat')
end

demoFolder = 'demoFolder';
if ~isdir(demoFolder)
    mkdir(demoFolder);
end
%%
figure(1); clf;
hold on;
grid on;
pbaspect([1 1 1]);
axis([-1. 1. -1. 1.]); % [xmin xmax ymin ymax zmin zmax]
xlabel('Valence');
ylabel('Arousal');
title('Two-Dimensional Emotion Model');

startingPoint = 60;
cutoffPoint = min(130, size(demoVideo.prediction,2)-1);
for frameIdx = startingPoint:cutoffPoint
    figure(1);
    ptrA = demoVideo.prediction(:,frameIdx);
    ptrB = demoVideo.prediction(:,frameIdx+1);
    line([ptrA(1);ptrB(1)],[ptrA(2);ptrB(2)],'linestyle','-','linewidth',2); % line([x1,x2],[y1,y2]); 
end
export_fig(fullfile(demoFolder, 'splashFigure_curve.eps'));

%% 
figure(1); 
% clf;
% hold on;
% grid on;
% pbaspect([1 1 1]);
% axis([-1. 1. -1. 1.]); % [xmin xmax ymin ymax zmin zmax]
% xlabel('Valence');
% ylabel('Arousal');
% title('Two-Dimensional Emotion Model');
% for frameIdx = startingPoint:cutoffPoint
%     figure(1);
%     ptrA = demoVideo.prediction(:,frameIdx);
%     ptrB = demoVideo.prediction(:,frameIdx+1);
%     line([ptrA(1);ptrB(1)],[ptrA(2);ptrB(2)],'linestyle','-','linewidth',2); % line([x1,x2],[y1,y2]); 
% end

for frameIdx = startingPoint:cutoffPoint
    if mod(frameIdx,5)==0
        figure(1);
        plot(demoVideo.prediction(1,frameIdx), demoVideo.prediction(2,frameIdx), 'r.', 'MarkerSize', 5);
        imwrite(uint8(demoVideo.frame(:,:,:,frameIdx)),...
            fullfile(demoFolder,...
            sprintf('frame%03d_V%.2f_A%.2f_D%.2f.png', frameIdx, curPrediction(1),curPrediction(2),curPrediction(3))));
    end
end

export_fig(fullfile(demoFolder, 'splashFigure_curve_withMarker.eps'));