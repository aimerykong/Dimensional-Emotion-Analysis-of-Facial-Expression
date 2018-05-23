%CNN_VGG_FACES  Demonstrates how to use VGG-Face
% Setup MatConvNet.
run(fullfile(fileparts(mfilename('fullpath')), ...
    './matconvnet', 'matlab', 'vl_setupnn.m')) ;

% Load the VGG-Face model.
% modelPath = fullfile(vl_rootnn,'data','models','vgg-face.mat') ;
modelPath = fullfile('/home/skong2/projects/basemodels','vgg-face.mat') ;
if ~exist(modelPath)
    fprintf('Downloading the VGG-Face model ... this may take a while\n') ;
    mkdir(fileparts(modelPath)) ;
    urlwrite(...
        'http://www.vlfeat.org/matconvnet/models/vgg-face.mat', ...
        modelPath) ;
end

% Load the model and upgrade it to MatConvNet current version.
net = load(modelPath) ;
net = vl_simplenn_tidy(net) ;

% Load a test image from Wikipedia and run the model.
im = imread('https://upload.wikimedia.org/wikipedia/commons/4/4a/Aamir_Khan_March_2015.jpg') ;
im = im(1:250,:,:) ; % crop
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
res = vl_simplenn(net, im_) ;

% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ; axis equal off ;
title(sprintf('%s (%d), score %.3f',...
    net.meta.classes.description{best}, best, bestScore), ...
    'Interpreter', 'none') ;

for i=1:length(net.layers)
    if strcmp(net.layers{i}.type,'conv')
        fprintf('%s\t%d  dim[%d %d %d %d]\n',net.layers{i}.name, length(net.layers{i}.weights),...
            size(net.layers{i}.weights{1},1), size(net.layers{i}.weights{1},2),...
            size(net.layers{i}.weights{1},3),  size(net.layers{i}.weights{1},4));
    else
        fprintf('%s\n',net.layers{i}.name);
    end
end
