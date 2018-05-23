
imgFig = figure(1); clf;
set(imgFig, 'Position', [10 10 1200 900]) % [1 1 width height]

% subplot(2,2,2);
% imshow(refPieChart);

frameIdx = 1;
subplot(2,2,1);
imshow(uint8(bsxfun(@plus,mat(:,:,:,frameIdx),meanVal)));
curPrediction = predLabel(:,frameIdx);
title(sprintf('V:%.3f, A:%.3f, D:%.3f',curPrediction(1),curPrediction(2),curPrediction(3)));

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
    elseif mod(frameIdx,4)==0 
        imwrite(imind, cm, gifName, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);
        imwrite(imindHighRes, cmHighRes, gifNameHighRes, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);        
    end
end
close(v);