function [psnr, ssim, fsim, error] = Img_QA(Img1, Img2)
%==========================================================================
% Evaluates the quality assessment indices for two RGB Imgaes.
%
% Syntax:
%   [psnr, ssim, fsim] = Img_QA(Img1, Img2)
%
% Input:
%   Img1 - the reference RGB image data array
%   Img2 - the target RGB image data array
% NOTE: RGB image data array is a M*N*3 array for imagery with M*N spatial
%	pixels with RGB channels in DYNAMIC RANGE [0, 255]. If Img1 and Img2
%	have different size, the larger one will be truncated to fit the smaller one.
%
% Output:
%   psnr - Peak Signal-to-Noise Ratio
%   ssim - Structure SIMilarity
%   fsim - Feature SIMilarity
%
%
% by Hailin Wang
%==========================================================================
psnr = PSNR(Img1, Img2, 1);
%img1 = 255*double(rgb2gray(Img1));
%img2 = 255*double(rgb2gray(Img2));
%ssim = ssim_index(max(Img1(:))*Img1, max(Img2(:))*Img2);
%fsim = FeatureSIM(max(Img1(:))*Img1, max(Img2(:))*Img2);


Nway = size(Img1);

ssim = zeros(prod(Nway(3:end)),1);
fsim = zeros(prod(Nway(3:end)),1);
for i = 1:prod(Nway(3:end))   
    %psnr(i) = psnr_index(Img1(:, :, i), Img2(:, :, i));
    ssim(i) = ssim_index(Img1(:, :, i), Img2(:, :, i));
    fsim(i) = FeatureSIM(Img1(:, :, i), Img2(:, :, i));
end
%psnr = nanmean(psnr);
ssim = nanmean(ssim);
fsim = nanmean(fsim);

error = norm(Img2(:)-Img1(:))/norm(Img1(:));

