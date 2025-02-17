% Demo code for debluring application.
%
% Author: Katherine Henneberger 
% 5/13/24

addpath(genpath(pwd))
clear;clc;close all;

load('data_foreman_video.mat');
%% process data 

num_frames = videoinfo.frameCount; %get the number of frames
scale_factor = 1;% define scaling factor if resizing video
X = [];
for i = 1:num_frames
        video_frame = mov(i).cdata;
        
        % Resize the video if desired
        video_frame = imresize(video_frame, scale_factor);

        X = cat(4, X, video_frame);
end
X= double(X); 
% cut off first few black columns and consider first 12 frames for a quick
% demo

X = X(:,4:176,:,1:12);
[N1,N2,N3,N4] = size(X);
h = fspecial('gaussian',[5,5],1);
[m2,n2] = size(h);

X3d = reshape(X,[N1,N2*N4,N3]);
[m1,n1,p] = size(X3d);
m = m1+m2-1;
n = n1+n2-1;

% generate a blurred video
A1 = conv4Dv4([m1,n1],h,[N1,N2,N3,N4]); % 3d blur kernel
H = decirc(A1,[size(A1,1)/N4,size(A1,2)/N4,m,N4]); % 4D blur kernel
%% 
Xt4d = zeros(N2,N3, m,N4 ); % num of frames. Width, frames, height
Xt4d(:,:,1:m1,:) = permute(X(end:-1:1,:,:,:),[2,3,1,4]);
Y3 = htprod_fft(H,Xt4d); % recover Y back to the same size
Y4 = Y3(:,:,end:-1:1,:); 
d1 = floor(m2/2); d2 = floor(n2/2);
%% 
Y4 = Y4(d2+1:N2+d2,:,d1+1:d1+m1,:);
Y4 = permute(Y4,[3,1,2,4]);
%% 
imshow(uint8(squeeze(Y4(:,:,:,5))))
%% 

it = 4;
for i = 1:it

    figure(i)
    imshow(uint8(squeeze(X(:,:,:,i))))
    %print(['pth_result_vid_orig_',num2str(i),''],'-depsc');
    
    figure(4+i)
    imshow(uint8(squeeze(Y4(:,:,:,i))))
    %print(['pth_result_vid_blur_',num2str(i),''],'-depsc');
    
    
end

%% set parameters
maxx = 1000;
para.maxit = maxx;
% if bs = 10 then full grad descent
para.gth = Xt4d;
para.alpha = 1;
para.lambda = .001;
para.p = 2;
para.bs = 80;
para.block = 3;
para.type = 'lowrank'; % try changing to sparse
para.control = 'cyc'; 
para.controltype = 'batch';

% run accelerated recovery algorithm
tic
out = pthtenrec_fft_4_accel(H,Y3,para);
time_prop = toc;
save('time.mat',"time_prop")
%% reshape
Xout = out.X; % recover back to the same size
Xrec = Xout(:,:,1:m1,:);
Xrec = Xrec(:,:,end:-1:1,:); 
Xrec = permute(Xrec,[3,1,2,4]);
Xrec(Xrec<0) = 0;
disp(norm(X-Xrec,'fro')/norm(X,'fro'))

%% 
figure(1)
imshow(uint8(squeeze(Xrec(:,:,:,5))))

figure(2)
imshow(uint8(squeeze(X(:,:,:,5))))

figure(3)
imshow(uint8(squeeze(Y4(:,:,:,5))))

%% visualize the results
it = 4;
for i = 1:it
    figure(i)
    imshow(uint8(squeeze(Xrec(:,:,:,i))))
end
%% print the relative error, psnr, fsim, ssim

psnr_index = PSNR(Xrec,X,255);
disp(psnr_index)
[~, ssim, fsim, ~] = Img_QA(X, Xrec);
fprintf('PSNR = %0.8e\n',psnr_index);
fprintf('SSIM = %0.8e\n',ssim);
fprintf('FSIM = %0.8e\n',fsim);

