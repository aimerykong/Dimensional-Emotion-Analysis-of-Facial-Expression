% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapper_TwoDimEmotion(opts)
% -------------------------------------------------------------------------
    fn = @(imdb,batch,mode) getBatch_TwoDimEmotion(imdb, batch, mode, opts) ;
end

% -------------------------------------------------------------------------

