% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapper_DimEmotion(opts)
% -------------------------------------------------------------------------
    fn = @(imdb,batch,mode) getBatch_DimEmotion(imdb, batch, mode, opts) ;
end

% -------------------------------------------------------------------------

