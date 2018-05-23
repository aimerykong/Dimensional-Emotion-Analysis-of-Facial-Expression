clear 
close all
clc;

addpath('exportFig');
%%
matFileList = {'result_main010_basemodel_v2_trainDimEmotion_resnet_L1_net-epoch-497.mat',...
    'result_main011_bilinearPool_v3_ftV1_fc1DimEmotion_resnet_L1_net-epoch-500.mat',...
    'result_main012_vggface2_v3_ftv1_1topConvDimEmotion_resnet_L1_net-epoch-500.mat',...
    'result_main013_vggface2_bilinear_v2_ftV1_fc1DimEmotion_resnet_L1_net-epoch-500.mat'};

figure(1);
matList = {};
thrList = 0:0.1:2;
markerStyle = {'^','s','o','+','x'};
colorList = {'r','b','m','k'};
for i = 1:length(matFileList)
    matList{end+1} = load([matFileList{i} '.mat']);    
    
    residuals = matList{end}.predLabel_test-matList{end}.grndLabel_test;
    RMSE = sqrt(mean(residuals.^2,1));    
    predErr_test = sqrt(residuals(:,1).^2/3 + residuals(:,2).^2/3 + residuals(:,3).^2/3);    
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

export_fig('assembledCurve_3DimModel.png', '-transparent');
