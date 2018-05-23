% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapper_DimEmotion_stage2(opts)
% -------------------------------------------------------------------------
    fn = @(imdb,batch,mode) getBatch_DimEmotion_stage2(imdb, batch, mode, opts) ;
end

% -------------------------------------------------------------------------

