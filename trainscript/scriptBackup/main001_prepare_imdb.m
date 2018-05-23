clear;
clc;
%%
load('./data/FacialExpressionRegression1.mat');
DEmotion = DEmotion/9;

TrainNum = 28561;
ValNum = 3579;
TestNum = 3574;
imsz = 48; % required by resnet input 224*224,original size is 48*48
%% train set
trainImages = zeros(imsz, imsz, 1, TrainNum);
trainEmotion = zeros(TrainNum, 3);
for i = 1:TrainNum    
    trainImages(:,:,1,i) = Images{i};
    trainEmotion(i,:) = DEmotion(i,1:3);
end
%% validation set
valImages = zeros(imsz,imsz,1,ValNum);
valEmotion = zeros(ValNum,3);
for i = TrainNum+1 : TrainNum+ValNum
    valImages(:,:,1,i-TrainNum) = Images{i};
    valEmotion(i-TrainNum,:) = DEmotion(i,1:3);
end
%% test set
testImages = zeros(imsz,imsz,1,TestNum);
testEmotion = zeros(TestNum,3);
for i = TrainNum+ValNum+1 : size(Images,1)
    testImages(:,:,1,i-TrainNum-ValNum) = Images{i};
    testEmotion(i-TrainNum-ValNum,:) = DEmotion(i,1:3);
end
%%
imdb.path = '/home/skong/data/feng_emotion/data/FacialExpressionRegression1.mat';
imdb.meta.project_name = 'DimEmotion';
imdb.train.image = trainImages;
imdb.train.annot = trainEmotion;
imdb.val.image = valImages;
imdb.val.annot = valEmotion;
imdb.test.image = testImages;
imdb.test.annot = testEmotion;

imdb.meta.mean_value = mean(trainImages(:));

save('imdb_DimEmotion.mat', 'imdb');