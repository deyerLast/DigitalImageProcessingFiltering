%Using fft to find the cracks of spatial domain
clear all;
close all;

% im = double(imread('./test/jf3.jpg'));
% im =(im(:,:,1)+im(:,:,2)+im(:,:,3))/3;
im =  imread('./test/jf3.jpg');
im=rgb2gray(im);
im2 = fftshift(fft2(im));
im3 = log(abs(im2)+1);% range of the data is too big
 
figure,imagesc(im);
colormap(gray(256));
axis image;
axis off;
title('original image');

figure,imagesc(im3);
colormap(gray(256));
axis image;
axis off;
title('In the frequency domain');

%% filter notch reject (ideal)
[r,c]=size(im2);
mask=ones(r,c);
for i=1:r
    for j=1:c
        if  i>423 && i<434 && j>0 && j< 290 
            mask(i,j)=0;
            im2(i,j)=mean(im2(:));
        end
         if  i>423 && i<434 && j>348 && j< 636
             mask(i,j)=0; 
             im2(i,j)=mean(im2(:));
        end
    end
end

figure,imshow(mask);
im4 = log(abs(im2)+1);

figure,imagesc(im4);colormap(gray(256));axis image;axis off;
nim = ifft2(im2);
nim = abs(nim);
figure,imshow(nim./255);
i=i;