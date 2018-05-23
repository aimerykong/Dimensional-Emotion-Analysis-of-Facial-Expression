function [imBatch, labelBatch] = getBatch_DimEmotion_stage1(imdb, batch, mode, opts)
% -------------------------------------------------------------------------

imBatch = zeros(imdb.meta.imagesize(1),imdb.meta.imagesize(2),3,length(batch),'single');
labelBatch = zeros(1,1,3,length(batch),'single');
if ~isempty(batch)
    for i = 1:length(batch)
        curImgIdx = batch(i);
        if strcmp(mode, 'train')
            im = imdb.train.image(:,:,:,curImgIdx);
            label = imdb.train.annot(curImgIdx,:);
        else
            im = imdb.val.image(:,:,:,curImgIdx);
            label = imdb.val.annot(curImgIdx,:);
        end
        %% randomly flip and rotate to augment training images --> 16 times larger in dataset scale
        if strcmp(mode, 'train')
            % random flipping
            if rand(1) > 0.5
                im = fliplr(im);
            end
        end
        %im = imresize(im, [224,224]);
        im = repmat(im, [1,1,3]);
        imBatch(:,:,:,i) = bsxfun(@minus, single(im), imdb.meta.mean_value);
        labelBatch(:,:,:,i) = single(reshape(label,[1 1 3]));        
    end
end
