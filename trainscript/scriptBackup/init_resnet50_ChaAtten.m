function net = init_resnet50_ChaAtten(n, insertBlockList, moduleStruct , varargin)

opts.batchNormalization = true;
opts.networkType = 'resnet'; % 'plain' | 'resnet'
opts.bottleneck = false; % only used when n is an array
opts.nClasses = 1000;
opts.reLUafterSum = true;
opts = vl_argparse(opts, varargin);
nClasses = opts.nClasses;


net = dagnn.DagNN();

% n -> specific configuration
if numel(n)==4,
    Ns = n;
else
    switch n,
        case 18, Ns = [2 2 2 2]; opts.bottleneck = false;
        case 34, Ns = [3 4 6 3]; opts.bottleneck = false;
        case 50, Ns = [3 4 6 3]; opts.bottleneck = true;
        case 101, Ns = [3 4 23 3]; opts.bottleneck = true;
        case 152, Ns = [3 8 36 3]; opts.bottleneck = true;
        otherwise, error('No configuration found for n=%d', n);
    end
end
if strcmpi(opts.networkType, 'plain') && opts.bottleneck,
    error('plain network cannot be built with bottleneck layers');
end

% Meta parameters
net.meta.inputSize = [48 48 1] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 256 ;
if opts.batchNormalization;
    net.meta.trainOpts.learningRate = [0.1*ones(1,30) 0.01*ones(1,30) 0.001*ones(1,50)] ;
else
    net.meta.trainOpts.learningRate = [0.01*ones(1,45) 0.001*ones(1,45) 0.0001*ones(1,75)] ;
end
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% First conv layer
switch n
    case {18, 34}
        block = dagnn.Conv('size',  [7 7 3 64], 'hasBias', true, ...
            'stride', 2, 'pad', [3 3 3 3]);
        lName = 'conv0';
        net.addLayer(lName, block, 'data', lName, {[lName '_f'], [lName '_b']});
    case {50, 101, 152}
        %% conv1
        block = dagnn.Conv('size',  [7 7 1 64], 'hasBias', false, 'stride', 1, 'pad', [3 3 3 3]);
        lName = 'conv1';
        net.addLayer(lName, block, 'data', lName, {[lName '_filter'], [lName '_bias']});
        sName = lName;
        if opts.batchNormalization
            lName = 'conv1_ChaAtten';
            add_layer_ChaAtten(net, 64, sName, lName, 0.0);
            sName = lName;
        end
        block = dagnn.ReLU('leak',0);
        net.addLayer('conv1_relu',  block, sName, 'conv1_relu');
        
%         %% conv1_1
%         block = dagnn.Conv('size',  [3 3 3 64], 'hasBias', false, 'stride', 2, 'pad', [1 1 1 1]);
%         lName = 'conv1_1';
%         net.addLayer(lName, block, 'data', lName, {[lName '_f']});
%         add_layer_ChaAtten(net, 64, lName, 'conv1_1_ChaAtten', 0.1);
%         block = dagnn.ReLU('leak',0);
%         net.addLayer('conv1_1_relu',  block, 'conv1_1_ChaAtten', 'conv1_1_relu');
%         %% conv1_2
%         block = dagnn.Conv('size',  [3 3 64 64], 'hasBias', false, 'stride', 1, 'pad', [1 1 1 1]);
%         lName = 'conv1_2';
%         net.addLayer(lName, block, 'conv1_1_relu', lName, {[lName '_f']});
%         add_layer_ChaAtten(net, 64, lName, 'conv1_2_ChaAtten', 0.1);
%         block = dagnn.ReLU('leak',0);
%         net.addLayer('conv1_2_relu',  block, 'conv1_2_ChaAtten', 'conv1_2_relu');
%         %% conv1_3
%         block = dagnn.Conv('size',  [3 3 64 128], 'hasBias', false, 'stride', 1, 'pad', [1 1 1 1]);
%         lName = 'conv1_3';
%         net.addLayer(lName, block, 'conv1_2_relu', lName, {[lName '_f']});
%         add_layer_ChaAtten(net, 128, lName, 'conv1_3_ChaAtten', 0.1);
%         block = dagnn.ReLU('leak',0);
%         net.addLayer('conv1_3_relu',  block, 'conv1_3_ChaAtten', 'conv1_3_relu');
end
% add_layer_bn(net, 64, lName, 'bn0', 0.1);
% block = dagnn.ReLU('leak',0);
% net.addLayer('relu0',  block, 'bn0', 'relu0');

%add_block_conv(net, '0', 'image', [7 7 3 64], 2, opts.batchNormalization, true);
% block = dagnn.Pooling('poolSize', [3 3], 'method', 'max', 'pad', [0 1 0 1], 'stride', 2);
% net.addLayer('pool1', block, 'conv1_relu', 'pool1');
%info.lastNumChannel = 128;
%info.lastName = 'pool1';

info.lastNumChannel = 64;
info.lastName = 'conv1_relu';
%% resBlock2
% Four groups of layers
curModuleStruct = [];
info.lastIdx = 1;
info.prefix = 'res2';
info.blockIdx = 1;
info = add_group(curModuleStruct, opts.networkType, net, Ns(1), info, 3, 64,  1, opts.bottleneck, opts.batchNormalization, opts.reLUafterSum);

%% resBlock3
flag_insert = find(insertBlockList==3);
if isempty(flag_insert)
    curModuleStruct = [];
else
    curModuleStruct = moduleStruct(flag_insert);
end
info.lastIdx = 1;
info.prefix = 'res3';
info.blockIdx = 1;
info.lastName = 'res2_3_relu';
info = add_group(curModuleStruct, opts.networkType, net, Ns(2), info, 3, 128, 2, opts.bottleneck, opts.batchNormalization, opts.reLUafterSum);

%% resBlock4
flag_insert = find(insertBlockList==4);
if isempty(flag_insert)
    curModuleStruct = [];
else
    curModuleStruct = moduleStruct(flag_insert);
end
info.lastIdx = 1;
info.prefix = 'res4';
info.blockIdx = 1;
info.lastName = 'res3_4_relu';
info = add_group(curModuleStruct, opts.networkType, net, Ns(3), info, 3, 256, 2, opts.bottleneck, opts.batchNormalization, opts.reLUafterSum);

%% resBlock5
flag_insert = find(insertBlockList==5);
if isempty(flag_insert)
    curModuleStruct = [];
else
    curModuleStruct = moduleStruct(flag_insert);
end
info.lastIdx = 1;
info.prefix = 'res5';
info.blockIdx = 1;
switch n
    case {50}
        info.lastName = 'res4_6_relu';
    case {101}
        info.lastName = 'res4_23_relu';
end
info = add_group(curModuleStruct, opts.networkType, net, Ns(4), info, 3, 512, 2, opts.bottleneck, opts.batchNormalization, opts.reLUafterSum);
%% Prediction & loss layers
block = dagnn.Pooling('poolSize', [7 7], 'method', 'avg', 'pad', 0, 'stride', 1);
if opts.reLUafterSum
    net.addLayer('pool_final', block, 'res5_3_relu', 'pool_final');
else
    net.addLayer('pool_final', block, 'res5_3_sum', 'pool_final');
end
block = dagnn.Conv('size', [1 1 info.lastNumChannel nClasses], 'hasBias', true, ...
    'stride', 1, 'pad', 0);
lName = sprintf('fc%d', info.lastIdx+1);
net.addLayer(lName, block, 'pool_final', lName, {[lName '_f'], [lName '_b']});


net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');
net.addLayer('loss', dagnn.Loss('loss', 'log'), {'softmax', 'label'}, 'loss');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax','label'}, 'error') ;
net.addLayer('error5', dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
    {'softmax','label'}, 'error5') ;

net.initParams();

net.meta.normalization.imageSize = net.meta.inputSize;
net.meta.normalization.border = 256 - net.meta.inputSize(1:2) ;
net.meta.normalization.interpolation = 'bicubic' ;
net.meta.normalization.averageImage = [] ;
net.meta.normalization.keepAspect = true ;
net.meta.augmentation.rgbVariance = zeros(0,3) ;
net.meta.augmentation.transformation = 'stretch' ;

%% Add a group of layers containing 2n/3n conv layers
function info = add_group(curModuleStruct, netType, net, n, info, w, ch, stride, bottleneck, bn, reLUafterSum)
if strcmpi(netType, 'plain'),
    if isfield(info, 'lastName'),
        lName = info.lastName;
        info = rmfield(info, 'lastName');
    else
        lName = sprintf('relu%d', info.lastIdx);
    end
    add_block_conv(net, sprintf('%d', info.lastIdx+1), lName, ...
        [w w info.lastNumChannel ch], stride, bn, true);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = ch;
    for i=2:2*n,
        add_block_conv(net, sprintf('%d', info.lastIdx+1), sprintf('relu%d', info.lastIdx), ...
            [w w ch ch], 1, bn, true);
        info.lastIdx = info.lastIdx + 1;
    end
elseif strcmpi(netType, 'resnet'),
    info = add_block_res([], net, info, [w w info.lastNumChannel ch], stride, bottleneck, bn, 1, reLUafterSum);
    for i=2:n,
        info.blockIdx = i;
        info.lastIdx = 1;
        if i ~= 2
            curModuleStruct = [];
        end
        if bottleneck,
            info = add_block_res(curModuleStruct, net, info, [w w 4*ch ch], 1, bottleneck, bn, 0, reLUafterSum);
        else
            info = add_block_res(curModuleStruct, net, info, [w w ch ch], 1, bottleneck, bn, 0, reLUafterSum);
        end
    end
end


%% Add a smallest residual unit (2/3 conv layers)
function info = add_block_res(curModuleStruct, net, info, f_size, stride, bottleneck, bn, isFirst, reLUafterSum)

if isfield(info, 'lastName') && isfield(info, 'prefix'),
    lName0 = info.lastName;
    info = rmfield(info, 'lastName');
elseif  isfield(info, 'lastName'),
    lName0 = info.lastName;
    info = rmfield(info, 'lastName');
elseif reLUafterSum || info.lastIdx == 0
    lName0 = sprintf('%s_%d_relu', info.prefix, info.blockIdx-1);
else
    lName0 = sprintf('%s_sum%d', info.prefix, info.lastIdx);
end

lName01 = lName0;
if stride > 1 || isFirst,
    if bottleneck,
        ch = 4*f_size(4);
    else
        ch = f_size(4);
    end
    block = dagnn.Conv('size',[1 1 f_size(3) ch], 'hasBias',false,'stride',stride, ...
        'pad', 0);
    lName_tmp = sprintf('%s_%d', info.prefix, info.blockIdx);
    lName_current = [lName_tmp '_projBranch'];
    net.addLayer(lName_current, block, lName0, lName_current, [lName_current '_f']);
    
    pidx = net.getParamIndex([lName_current '_f']);
    net.params(pidx).learningRate = 1;
    
    %add_layer_bn(net, ch, lName_current, [lName_tmp '_projBranch_bn'], 0.1);
    %lName0 = [lName_tmp '_projBranch_bn'];
    
    add_layer_ChaAtten(net, ch, lName_current, [lName_tmp '_projBranch_ChaAtten'], 0.0);
    lName0 = [lName_tmp '_projBranch_ChaAtten'];
end


if bottleneck,
    add_block_conv(net, sprintf('%s_%d_%d', info.prefix, info.blockIdx, info.lastIdx), lName01, [1 1 f_size(3) f_size(4)], stride, bn, true);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = f_size(4);
    
    %% here is the scale pooling pyramid module!!!
%     out = sprintf('%s_%d_%d', info.prefix, info.blockIdx, info.lastIdx)
%     in = sprintf('%s_%d_%drelu', info.prefix, info.blockIdx, info.lastIdx-1)
    if  ~isempty(curModuleStruct)      
        in_name = sprintf('%s_%d_%drelu', info.prefix, info.blockIdx, info.lastIdx-1);
        out_suffix = sprintf('%s_%d_%d', info.prefix, info.blockIdx, info.lastIdx);
        namePrefix = sprintf('ScaleAttention_%s', out_suffix);
        
        [net, out_suffix] = insertScaleAttentionPyramidModule(net, info.lastNumChannel, curModuleStruct.poolScaleList, in_name, namePrefix);
        info.lastIdx = info.lastIdx + 1;
        
        in_name = out_suffix;
    else
        out_suffix = sprintf('%s_%d_%d', info.prefix, info.blockIdx, info.lastIdx);
        in_name = sprintf('%s_%d_%drelu', info.prefix, info.blockIdx, info.lastIdx-1);
        add_block_conv(net, ...
            out_suffix, ...
            in_name, ...
            [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1, bn, true);
        info.lastIdx = info.lastIdx + 1;        
        in_name = sprintf('%s_%d_%drelu', info.prefix, info.blockIdx, info.lastIdx-1);
    end
    %%    
    out_suffix = sprintf('%s_%d_%d', info.prefix, info.blockIdx, info.lastIdx);
    add_block_conv(net, ...
        out_suffix, ...
        in_name, ...
        [1 1 info.lastNumChannel info.lastNumChannel*4], 1, bn, false);
    %     info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = info.lastNumChannel*4;
else
    add_block_conv(net, sprintf('%s_%d',  info.prefix, info.lastIdx+1), lName01, f_size, stride, bn, true);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = f_size(4);
    add_block_conv(net, sprintf('%s_%d', info.prefix, info.lastIdx+1), sprintf('%s_relu%d',  info.prefix, info.lastIdx), ...
        [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1, bn, true);
    info.lastIdx = info.lastIdx + 1;
end

if bn,
    %     lName1 = sprintf('%s_%d_3bn',  info.prefix, info.blockIdx);
    lName1 = sprintf('%s_%d_3ChaAtten',  info.prefix, info.blockIdx);
else
    lName1 = sprintf('%s_%d_conv',  info.prefix, info.blockIdx);
end

info.lastIdx = info.lastIdx + 1;
net.addLayer(sprintf('%s_%d_sum', info.prefix, info.blockIdx), dagnn.Sum(), {lName0,lName1}, ...
    sprintf('%s_%d_sum', info.prefix, info.blockIdx));

% relu
if reLUafterSum
    block = dagnn.ReLU('leak', 0);
    net.addLayer(sprintf('%s_%d_relu', info.prefix, info.blockIdx), block, sprintf('%s_%d_sum',  info.prefix, info.blockIdx), ...
        sprintf('%s_%d_relu',  info.prefix, info.blockIdx));
end


%% Add a conv layer (followed by optional batch normalization & relu)
function net = add_block_conv(net, out_suffix, in_name, f_size, stride, bn, relu)
block = dagnn.Conv('size',f_size, 'hasBias',false, 'stride', stride, ...
    'pad',[ceil(f_size(1)/2-0.5) floor(f_size(1)/2-0.5) ...
    ]);
lName = [ out_suffix 'conv'];
net.addLayer(lName, block, in_name, lName, {[lName '_f']});

if bn,
    %add_layer_bn(net, f_size(4), lName, strrep(lName,'conv','bn'), 0.1);
    add_layer_ChaAtten(net, f_size(4), lName, strrep(lName,'conv','ChaAtten'), 0.0);
    %lName = strrep(lName, 'conv', 'bn');
    lName = strrep(lName, 'conv', 'ChaAtten');
end
if relu,
    block = dagnn.ReLU('leak',0);
    net.addLayer([out_suffix 'relu'], block, lName, [out_suffix 'relu']);
end


%% Add a batch normalization layer
function net = add_layer_bn(net, n_ch, in_name, out_name, lr)
block = dagnn.BatchNorm('numChannels', n_ch);
net.addLayer(out_name, block, in_name, out_name, ...
    {[out_name '_g'], [out_name '_b'], [out_name '_m']});
pidx = net.getParamIndex({[out_name '_g'], [out_name '_b'], [out_name '_m']});
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(3)).learningRate = lr;
net.params(pidx(3)).trainMethod = 'average';


%% Add a Channel-wise Attention layer (ChaAtten)
function net = add_layer_ChaAtten(net, n_ch, in_name, out_name, lr)
block = attentionChannel('numChannels', n_ch);
net.addLayer(out_name, block, in_name, out_name, ...
    {[out_name '_multiplier'], [out_name '_bias']});
pidx = net.getParamIndex({[out_name '_multiplier'], [out_name '_bias']});
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(1)).learningRate = lr;
net.params(pidx(2)).learningRate = lr;
%net.params(pidx(3)).trainMethod = 'average'; % gradient or average

