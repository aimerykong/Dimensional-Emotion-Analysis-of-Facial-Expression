% quick setup
clear;
close all;
clc;

addpath(genpath('../libs'))
path_to_matconvnet = '/home/skong2/local/depthAwareSegTransfer/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
path_to_model = '~/local/basemodels/';

load('imdb_DimEmotion.mat');

gpuId = 3;
gpuDevice(gpuId);
%%
scalingFactor = 1;
opts.batchNormalization = 1 ;
opts.nClasses = 10;
opts.networkType = 'resnet' ;

depth = 50;

insertBlockList = [];
poolScaleList = [1 2 4 6 8 10];
moduleStruct(1).poolScaleList = poolScaleList;
moduleStruct(1).inDim = 512;
moduleStruct(1).namePrefix = 'res5attention';
moduleStruct(1).sName = 'res5_1_relu';

netbasemodel = init_resnet50_ChaAtten(depth, insertBlockList, moduleStruct, 'nClasses', opts.nClasses, ...
                        'batchNormalization', opts.batchNormalization, ...
                        'networkType', opts.networkType) ;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.removeLayer('error5'); % remove layer
netbasemodel.removeLayer('error'); % remove layer
netbasemodel.removeLayer('loss'); % remove layer
netbasemodel.removeLayer('softmax'); % remove layer
netbasemodel.removeLayer('fc5'); % remove layer
netbasemodel.removeLayer('pool_final'); % remove layer
%%
% initNet.move('gpu');
% %% add new layers on resblock5 to train from scratch
% sName = 'res5_3_relu';
% lName = 'res6_conv';
% FoV_size = 1;
% outDim = 512;
% block = dagnn.Conv('size', [3 3 2048 outDim], 'hasBias', false, 'stride', 1, 'pad', FoV_size, 'dilate', FoV_size);
% netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f']});
% ind = netbasemodel.getParamIndex([lName '_f']);
% weights = randn(3, 3, 2048, outDim, 'single')*sqrt(2/outDim);
% netbasemodel.params(ind).value = weights;
% netbasemodel.params(ind).weightDecay = 1;
% netbasemodel.params(ind).learningRate = 10;
% sName = lName;
% inputDim = outDim;
% 
% lName = 'res6_ChaAtten';
% block = attentionChannel('numChannels', inputDim);
% netbasemodel.addLayer(lName, block, sName, lName, {[lName '_multiplier'], [lName '_bias']});
% pidx = netbasemodel.getParamIndex({[lName '_multiplier'], [lName '_bias']});
% netbasemodel.params(pidx(1)).weightDecay = 1;
% netbasemodel.params(pidx(1)).learningRate = 1;
% netbasemodel.params(pidx(2)).weightDecay = 1;
% netbasemodel.params(pidx(2)).learningRate = 1;
% netbasemodel.params(pidx(1)).value = ones(512, 1, 'single'); % slope
% netbasemodel.params(pidx(2)).value = zeros(512, 1, 'single');  % bias
% sName = lName;
% 
% lName = 'res6_relu';
% block = dagnn.ReLU('leak', 0);
% netbasemodel.addLayer(lName, block, sName, lName);
% sName = lName;
% 
% 
% lName = 'res7_conv';
% FoV_size = 1;
% outDim = 512;
% block = dagnn.Conv('size', [3 3 outDim outDim], 'hasBias', false, 'stride', 1, 'pad', FoV_size, 'dilate', FoV_size);
% netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f']});
% ind = netbasemodel.getParamIndex([lName '_f']);
% weights = randn(3, 3, outDim, outDim, 'single')*sqrt(2/outDim);
% netbasemodel.params(ind).value = weights;
% netbasemodel.params(ind).weightDecay = 1;
% netbasemodel.params(ind).learningRate = 10;
% sName = lName;
% inputDim = outDim;
% 
% lName = 'res7_ChaAtten';
% block = attentionChannel('numChannels', inputDim);
% netbasemodel.addLayer(lName, block, sName, lName, {[lName '_multiplier'], [lName '_bias']});
% pidx = netbasemodel.getParamIndex({[lName '_multiplier'], [lName '_bias']});
% netbasemodel.params(pidx(1)).weightDecay = 1;
% netbasemodel.params(pidx(1)).learningRate = 1;
% netbasemodel.params(pidx(2)).weightDecay = 1;
% netbasemodel.params(pidx(2)).learningRate = 1;
% netbasemodel.params(pidx(1)).value = ones(512, 1, 'single'); % slope
% netbasemodel.params(pidx(2)).value = zeros(512, 1, 'single');  % bias
% sName = lName;
% 
% lName = 'res7_relu';
% block = dagnn.ReLU('leak', 0);
% netbasemodel.addLayer(lName, block, sName, lName);
% sName = lName;
% 
% 
% lName = 'res7_dropout' ;
% block = dagnn.DropOut('rate', 0.5);
% netbasemodel.addLayer(lName, block, sName, lName);
% sName = lName;
% 
% 
% block = dagnn.Conv('size', [1 1 inputDim imdb.meta.classNum], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
% lName = 'res8_conv';
% netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
% ind = netbasemodel.getParamIndex([lName '_f']);
% weights = randn(1, 1, 512, imdb.meta.classNum, 'single')*sqrt(2/imdb.meta.classNum);
% netbasemodel.params(ind).value = weights;
% netbasemodel.params(ind).learningRate = 10;
% ind = netbasemodel.getParamIndex([lName '_b']);
% netbasemodel.params(ind).value = zeros([1 imdb.meta.classNum], 'single');
% netbasemodel.params(ind).learningRate = 20;
% sName = lName;
% 
% baseName = 'res8';
% upsample_fac = 8;
% filters = single(bilinear_u(upsample_fac*2, imdb.meta.classNum, imdb.meta.classNum));
% crop = ones(1,4) * upsample_fac/2;
% deconv_name = [baseName, '_interp'];
% var_to_up_sample = sName;
% netbasemodel.addLayer(deconv_name, ...
%     dagnn.ConvTranspose('size', size(filters), ...
%     'upsample', upsample_fac, ...
%     'crop', crop, ...
%     'opts', {'cudnn','nocudnn'}, ...
%     'numGroups', imdb.meta.classNum, ...
%     'hasBias', false), ...
%     var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
% ind = netbasemodel.getParamIndex([deconv_name  '_f']) ;
% netbasemodel.params(ind).value = filters ;
% netbasemodel.params(ind).learningRate = 0 ;
% netbasemodel.params(ind).weightDecay = 1 ;
% sName = deconv_name;
% 
% 
% obj_name = sprintf('obj_div%d_seg', scalingFactor);
% gt_name =  sprintf('gt_div%d_seg', scalingFactor);
% input_name = sName;
% netbasemodel.addLayer(obj_name, ...
%     SegmentationLossLogistic('loss', 'softmaxlog'), ... softmaxlog logistic
%     {input_name, gt_name}, obj_name)
% 
% 
% % modify res3's pooling size
% % netbasemodel.layers(39).block.stride = [1 1]; %res3a_branch2a
% % netbasemodel.layers(47).block.stride = [1 1]; %res3a_branch1
% 
% % ResBlock4
% % netbasemodel.layers(55).block.stride = [1 1]; % res4_1_projBranch
% % netbasemodel.layers(56).block.stride = [1 1]; % res4_1_1conv
% netbasemodel.layers(85).block.stride = [1 1]; % res4_1_projBranch
% netbasemodel.layers(87).block.stride = [1 1]; % res4_1_1conv
% dilationRate = 2;
% for idx = 1:length(netbasemodel.layers)
%     if ~isempty(strfind(netbasemodel.layers(idx).name, 'res4')) ... 
%             && isempty(strfind(netbasemodel.layers(idx).name, 'ScaleAttention')) ...  
%             && strcmpi( class(netbasemodel.layers(idx).block), 'dagnn.Conv' ) ...  
%             && isempty(strfind(netbasemodel.layers(idx).name, 'relu')) ...          
%             && netbasemodel.layers(idx).block.size(1) == 3  
%         netbasemodel.layers(idx).block.dilate = [dilationRate dilationRate];
%         netbasemodel.layers(idx).block.pad = [dilationRate dilationRate];
%         disp(netbasemodel.layers(idx).name);
%     end
% end
% 
% % ResBlock5
% netbasemodel.layers(147).block.stride = [1 1]; % res5_1_projBranch
% netbasemodel.layers(149).block.stride = [1 1]; % res5_1_1conv
% dilationRate = 4;
% for idx = 1:numel(netbasemodel.layers)    
%     if ~isempty(strfind(netbasemodel.layers(idx).name, 'res5')) ...       
%             && isempty(strfind(netbasemodel.layers(idx).name, 'ScaleAttention')) ...       
%             && strcmpi( class(netbasemodel.layers(idx).block), 'dagnn.Conv' ) ...  
%             && isempty(strfind(netbasemodel.layers(idx).name, 'relu')) ...
%             && netbasemodel.layers(idx).block.size(1) == 3   
%         netbasemodel.layers(idx).block.dilate = [dilationRate dilationRate];
%         netbasemodel.layers(idx).block.pad = [dilationRate dilationRate];
%         disp(netbasemodel.layers(idx).name);
%     end
% end
%% reference model pretrained on imagenet
weightbasemodel = load(fullfile(path_to_model, 'imagenet-resnet-50-dag.mat'));
for i = 1:length(weightbasemodel.layers)
    if isfield(weightbasemodel.layers(i).block, 'bnorm_moment_type_trn')
        weightbasemodel.layers(i).block = rmfield(weightbasemodel.layers(i).block, 'noise_param_idx');
        weightbasemodel.layers(i).block = rmfield(weightbasemodel.layers(i).block, 'noise_cache_size');
        weightbasemodel.layers(i).block = rmfield(weightbasemodel.layers(i).block, 'bnorm_moment_type_trn');
        weightbasemodel.layers(i).block = rmfield(weightbasemodel.layers(i).block, 'bnorm_moment_type_tst');
    end
end
weightbasemodel = dagnn.DagNN.loadobj(weightbasemodel);
weightbasemodel.removeLayer('prob'); % remove layer
weightbasemodel.removeLayer('fc1000'); % remove layer
weightbasemodel.removeLayer('pool5'); % remove layer
%% construct corresponding convolution layers
layerNameList_init = {};
for i = 1:length(netbasemodel.layers)
    if strcmpi( class(netbasemodel.layers(i).block), 'dagnn.Conv' )
        fprintf('''%s'', \n', netbasemodel.layers(i).name);
        layerNameList_init{end+1} = netbasemodel.layers(i).name;
    end
end

layerNameList_weight = {};
for i = 1:length(weightbasemodel.layers)
    if strcmpi( class(weightbasemodel.layers(i).block), 'dagnn.Conv' )
        fprintf('''%s'', \n', weightbasemodel.layers(i).name);
        layerNameList_weight{end+1} = weightbasemodel.layers(i).name;
    end
end


for i = 1:length(layerNameList_weight)
    fprintf('%s\t%s\n', layerNameList_init{i}, layerNameList_weight{i});
end
%% copy the weights from convolution layers
ind = netbasemodel.getParamIndex('conv1_filter');
weights = randn(7, 7, 1, 64, 'single')*sqrt(2/64);
netbasemodel.params(ind).value = weights;

ind = netbasemodel.getParamIndex('conv1_bias');
netbasemodel.params(ind).value = zeros([1 64], 'single');

targetLayerIdx = 2;
count_in_a_row_ScalePoolPyramidModule = 0;
for sourceLayerIdx = 2:length(layerNameList_init)
    currentLayerName_init = layerNameList_init{sourceLayerIdx};
    layerID_init = netbasemodel.getLayerIndex(currentLayerName_init);
    paramsNames_init = netbasemodel.layers(layerID_init).params;
    
    if contains(currentLayerName_init, 'ScaleAttention') &&  contains(currentLayerName_init, '_pyramid_pool')
        count_in_a_row_ScalePoolPyramidModule = count_in_a_row_ScalePoolPyramidModule + 1;
        if count_in_a_row_ScalePoolPyramidModule>1
            targetLayerIdx = targetLayerIdx - 1;
        end
    elseif contains(currentLayerName_init, '_AttentionLayer')
        continue;     
    else
        if count_in_a_row_ScalePoolPyramidModule > 0 
            count_in_a_row_ScalePoolPyramidModule = 0;
        end
    end
    
    currentLayerName_weight = layerNameList_weight{targetLayerIdx};
    targetLayerIdx = targetLayerIdx + 1;
    
    layerID_weight = weightbasemodel.getLayerIndex(currentLayerName_weight);
    paramsNames_weight = weightbasemodel.layers(layerID_weight).params;
    for paramIdx = 1:length(paramsNames_init)
        cur_param_init = paramsNames_init{paramIdx};
        cur_param_weight = paramsNames_weight{paramIdx};
        
        cur_param_init = netbasemodel.getParamIndex(cur_param_init);
        cur_param_weight = weightbasemodel.getParamIndex(cur_param_weight);
        netbasemodel.params(cur_param_init).value = weightbasemodel.params(cur_param_weight).value ;
    end
    
end
%% construct corresponding batch normalization layers
layerNameList_init = {};
for i = 1:length(netbasemodel.layers)
    if strcmpi( class(netbasemodel.layers(i).block), 'attentionChannel' )
        fprintf('''%s'', \n', netbasemodel.layers(i).name);
        layerNameList_init{end+1} = netbasemodel.layers(i).name;
    end
end

layerNameList_weight = {};
for i = 1:length(weightbasemodel.layers)
    if strcmpi( class(weightbasemodel.layers(i).block), 'dagnn.BatchNorm' )
        fprintf('''%s'', \n', weightbasemodel.layers(i).name);
        layerNameList_weight{end+1} = weightbasemodel.layers(i).name;
    end
end


for i = 1:length(layerNameList_weight)
    fprintf('%s\t%s\n', layerNameList_init{i}, layerNameList_weight{i});
end
%% copy the weights from batch normalization layers
sourceLayerIdx = 1;
targetLayerIdx = 1;
count_in_a_row_ScalePoolPyramidModule = 0;
for sourceLayerIdx = 1:length(layerNameList_init)
    currentLayerName_init = layerNameList_init{sourceLayerIdx};
    layerID_init = netbasemodel.getLayerIndex(currentLayerName_init);
    paramsNames_init = netbasemodel.layers(layerID_init).params;
    
    if contains(currentLayerName_init, 'ScaleAttention') &&  contains(currentLayerName_init, '_pyramid_pool')
        count_in_a_row_ScalePoolPyramidModule = count_in_a_row_ScalePoolPyramidModule + 1;
        if count_in_a_row_ScalePoolPyramidModule>1
            targetLayerIdx = targetLayerIdx - 1;
        end
    elseif contains(currentLayerName_init, '_AttentionLayer')
        continue;
    else
        if count_in_a_row_ScalePoolPyramidModule > 0
            count_in_a_row_ScalePoolPyramidModule = 0;
        end
    end
    
    currentLayerName_weight = layerNameList_weight{targetLayerIdx};
    targetLayerIdx = targetLayerIdx + 1;
    
    layerID_weight = weightbasemodel.getLayerIndex(currentLayerName_weight);
    paramsNames_weight = weightbasemodel.layers(layerID_weight).params;    
    %% get value from Batch Normalization
    bn_conv1_mult = paramsNames_weight{1};
    bn_conv1_mult = weightbasemodel.getParamIndex(bn_conv1_mult);
    bn_conv1_mult = weightbasemodel.params(bn_conv1_mult).value;
    bn_conv1_bias = paramsNames_weight{2};
    bn_conv1_bias = weightbasemodel.getParamIndex(bn_conv1_bias);
    bn_conv1_bias = weightbasemodel.params(bn_conv1_bias).value;
    bn_conv1_moments = paramsNames_weight{3};
    bn_conv1_moments = weightbasemodel.getParamIndex(bn_conv1_moments);
    bn_conv1_moments = weightbasemodel.params(bn_conv1_moments).value;
        
    multiplicative_const = bn_conv1_mult;
    additive_const = bn_conv1_bias;
    globalMean = bn_conv1_moments(:,1);
    globalVariance = bn_conv1_moments(:,2);
    
    epsilon = 0.0001;
    multiplier = multiplicative_const ./ (globalVariance + epsilon);
    bias = additive_const - multiplier.* globalMean;    
    
    fprintf('%s \t %s:', currentLayerName_init, currentLayerName_weight);
    fprintf('maxMult:%.2f, maxBias:%.2f\n', ...
        max(bn_conv1_mult), max(bn_conv1_bias));
    fprintf('\tmaxGlobalMean:%.2f, maxGlobalVariance:%.2f\n', ...
        max(globalMean), max(globalVariance));
    %% copy to ChaAtten 
    netbasemodel.layers(layerID_init).block.globalMean = globalMean;
    netbasemodel.layers(layerID_init).block.globalVariance = globalVariance;
    
    cur_param_init = paramsNames_init{1};
    cur_param_init = netbasemodel.getParamIndex(cur_param_init);
    netbasemodel.params(cur_param_init).value = bn_conv1_mult; 
    %netbasemodel.params(cur_param_init).value = multiplier;    
    %netbasemodel.params(cur_param_init).value = [bn_conv1_mult, bn_conv1_bias];
    
    cur_param_init = paramsNames_init{2};
    cur_param_init = netbasemodel.getParamIndex(cur_param_init);
    netbasemodel.params(cur_param_init).value = bn_conv1_bias;   
    %netbasemodel.params(cur_param_init).value = bias;   
    %netbasemodel.params(cur_param_init).value = bn_conv1_moments;    
end
%%
%disp({netbasemodel.layers.name}');

%% leaving blank
netbasemodel = netbasemodel.saveobj() ;
save('init_resnet50_ChaAtten.mat', '-struct', 'netbasemodel') ;
