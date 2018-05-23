clear 
close all
clc;

addpath('exportFig');
%%
matFileList = {'result_main014_2dimModel_vgg16_v1_forwardDimEmotion_resnet_L1_net-epoch-100.mat',...
    'result_main014_2dimModel_vgg16_v2_bilinearDimEmotion_resnet_L1_net-epoch-100.mat',...
    'result_main015_2dimModel_res50_v1_forwardDimEmotion_resnet_L1_net-epoch-100.mat',...
    'result_main015_2dimModel_res50_v2_bilinearDimEmotion_resnet_L1_net-epoch-54.mat'};

figure(1);
matList = {};
thrList = 0:0.1:2;
markerStyle = {'^','s','o','+','x'};
colorList = {'r','b','m','k'};
for i = 1:length(matFileList)
    matList{end+1} = load([matFileList{i} '.mat']);    
    
    residuals = matList{end}.predLabel_test-matList{end}.grndLabel_test(:,1:2);
    RMSE = sqrt(mean(residuals.^2,1));    
    predErr_test = sqrt(residuals(:,1).^2/2 + residuals(:,2).^2/2 );    
    squares = predErr_test.^2;
    rmse = sqrt(mean(squares));
        
    accList = [];
    for thr = thrList
        numCorrect = sum(abs(predErr_test)<thr);
        accuracy = numCorrect/numel(predErr_test);
        accList(end+1) = accuracy;
    end
    plot(thrList, accList, [colorList{i} '-' markerStyle{i}]);
    hold on;
    disp(accList(find(thrList==1)))
    
end
lgd = legend({'VGG16', 'VGG16-bilinear', 'ResNet50', 'ResNet50-bilinear'}, ...
    'Location', 'southeast', 'FontSize', 12);

export_fig('assembledCurve_2DimModel.png', '-transparent');
