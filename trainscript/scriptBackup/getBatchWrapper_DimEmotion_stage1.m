% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapper_DimEmotion_stage1(opts)
% -------------------------------------------------------------------------
    fn = @(imdb,batch,mode) getBatch_DimEmotion_stage1(imdb, batch, mode, opts) ;
end

% -------------------------------------------------------------------------

