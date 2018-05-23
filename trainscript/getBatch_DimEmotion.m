function [imBatch, labelBatch] = getBatch_DimEmotion(imdb, batch, mode, opts)
% -------------------------------------------------------------------------

imBatch = zeros(imdb.meta.imagesize(1),imdb.meta.imagesize(2),imdb.meta.imagesize(3),length(batch),'single');
labelBatch = zeros(1,1,3,length(batch),'single');
if ~isempty(batch)
    for i = 1:length(batch)
        curImgIdx = batch(i);
        if strcmp(mode, 'train')
            im = imdb.train.image(:,:,:,curImgIdx);
            label = imdb.train.annot(curImgIdx,:);
            
            chance=0.5;
            border=5;
            ystart = 1;
            xstart = 1;
            yend = 1;
            xend = 1;
            if rand(1)<chance
                xstart = randperm(border,1);
            end
            if rand(1)<chance
                xend = randperm(border,1)-1;
            end
            if rand(1)<chance
                ystart = randperm(border,1);
            end
            if rand(1)<chance
                yend = randperm(border,1)-1;
            end
            im = im(ystart:end-yend, xstart:end-xend);
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
        im = imresize(im, [size(imBatch,1),size(imBatch,2)]);
        if size(im,3)<imdb.meta.imagesize(3)
            im = repmat(im, [1,1,imdb.meta.imagesize(3)]);
        end
        imBatch(:,:,:,i) = bsxfun(@minus, single(im), imdb.meta.mean_value);
        labelBatch(:,:,:,i) = single(reshape(label,[1 1 3]));        
    end
end
