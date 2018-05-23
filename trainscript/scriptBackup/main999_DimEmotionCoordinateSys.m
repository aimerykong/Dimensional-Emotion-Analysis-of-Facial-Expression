clear; close all; clc;

demoFolder = 'demoFolder';
if ~isdir(demoFolder)
    mkdir(demoFolder);
end
%%
[x,z] = meshgrid(-1:0.005:1, -1:0.005:1);
z=-z;
M = x.^2+z.^2;
M = M<=1;
x = x.*M;
z = z.*M;
y = sqrt(abs(1-x.^2-z.^2));
y = y.*M;
im = cat(3,-x,z,y);

im = flipud(im);
im = fliplr(im);

imDemo = bsxfun(@times,(im/5+0.8), M);
imDemo = imDemo./max(imDemo(:));

A = sum(abs(imDemo),3);
A = find(A==0);
for i = 1:3
    tmp = imDemo(:,:,i);
    tmp(A) = 1;
    imDemo(:,:,i) = tmp;
end

addpath('exportFig');
imagesc(imDemo); axis image off;
export_fig(fullfile(demoFolder, 'colorRefPieChart.eps'));