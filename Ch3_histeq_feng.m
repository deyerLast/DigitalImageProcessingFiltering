%This program is to equalize the histogram for image enhancement
clear all
close all
clc
im=imread('./iris.bmp');%woman_blonde.tif
figure,imshow(im) 
%% imhist and histeq
im_eq = histeq(im);
figure,imshow(im_eq) 
figure,imhist(im)
figure,imhist(im_eq)


% %% imadjust
% ima=imadjust(im,[],[],0.3)
% figure,imshow(ima)


%% %%%%%%%% user version function
im=double(im);
c=myhist(im); %User our own function to generate the histogram

d=zeros(256,1);
[co,ro]=size(im);

old_hist = c;
for i=2:256
   c(i)=c(i)+c(i-1);
end

c=c*255;
% cinv=zeros(255,1);
% for i=1:255
%      cinv(floor(c(i)+1))=i; 
% end
figure,plot(c)
%figure,plot(cinv)


for x=1:co
   for y=1:ro
      f(x,y)=floor(c(im(x,y)+1));
   end
end

new_hist = myhist(f);

figure;
subplot(2,1,1),image(im);
colormap(gray(256));
title('original image');
axis image;
axis off;
subplot(2,1,2),bar(old_hist);

figure;
subplot(2,1,1),image(f);
colormap(gray(256));
axis image;
axis off;
subplot(2,1,2),bar(new_hist);