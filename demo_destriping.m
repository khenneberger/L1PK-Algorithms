% Demo code for destriping application.
%
% Author: Katherine Henneberger 
% 5/13/24


clear
addpath(genpath(pwd));
 
%%  Load Face data
load('YaleFace.mat');
X = YaleFace./max(YaleFace(:));
[d1, d2 ,d3,d4] = size(X);
maxP = max(abs(X(:)));
maxP1 = max(YaleFace(:));
Nways=size(X);


%%  shift dims
% create A
A = zeros(d1,d1,d3,d4);
maxdiag = d1;
% create facewise diagonal tensor
for i = 1:maxdiag
    A(i,i,:,:) = 1;
end
% create stripe
for i = 5:5:48
    A(i,i,:,:)=.01;
end
for i = 3:4
    A = ifft(A,[],i);
end

%% 

XT = X; 
[n1, n2 ,n3,n4] = size(XT);

%% initial parameters
Y = htprod_fft(A,XT);
para.maxit = 48; 
para.bs = 1; 
para.block = 2;
para.control = 'cyc'; 
para.type = 'lowrank';
para.tol = 1e-2;
para.gth = XT;
para.controltype = 'batch';
para.lambda = .001;
para.p = 1; %1 was best
para.alpha = 1;
para.gth = XT;

%% Fast  Fouier  Transform (FFT)
  fprintf('===== Recovery by FFT =====\n');
 
   t0=tic;
     out = pthtenrec_fft_4(A,Y,para);
   time = toc(t0);
   %% 
   Xhat1=max(0,out.X);
   Xhat2=min(maxP,Xhat1);
   err = out.err;
%% print the relative error, psnr, fsim, ssim
    Error = norm(Xhat2(:)-X(:))/norm(X(:));
    fprintf('Relative error = %0.8e\n',Error);
    psnr_index = PSNR(Xhat2,X,maxP);
    [~, ssim, fsim, ~] = Img_QA(X, Xhat2);
    fprintf('PSNR = %0.8e\n',psnr_index);
    fprintf('SSIM = %0.8e\n',ssim);
    fprintf('FSIM = %0.8e\n',fsim);

%% visualize the results

figure(1);
imagesc(Xhat2(:,:,11,10)); axis off; colormap gray; axis image;

figure(2);
imagesc(Xhat2(:,:,11,13)); axis off; colormap gray; axis image;

figure(3);
imagesc(Xhat2(:,:,11,12)); axis off; colormap gray; axis image;


