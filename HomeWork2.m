%David Meyer
%%Homework 2: Filters and Set Boundary

clc 
clear all
close all
%%
picToRead='./sobel.png'; % path of the image
%pic=double(imread(picToRead));
%read in image and convert it to double format
pic = im2double(imread(picToRead));
%pic = pic(:,:,1);
figure,imshow(pic)

%Change to GrayScale
[~, columns, numberOfColorChannels] = size(pic);
if numberOfColorChannels > 1
    % It's a true color RGB image.  We need to convert to gray scale.
    grayPic = rgb2gray(pic);
else
    % It's already gray scale.  No need to convert.
    grayPic = pic;
    fprintf("already gray")
end
figure,imshow(grayPic)

figure,imshow(avgFilter(grayPic))
picSalt = imnoise(grayPic, 'salt & pepper', 0.06); %matlab function to add salt & pepper noise.
figure,montage({picSalt,medFilt(grayPic)})
%imsave(montage({picSalt,medFilt(grayPic)})) % GOOGLE docs doesn't take
%this format, and it's to late for me to continue working on this.
%JPEG not offered for this while saving
figure,imshow(sobFilter(grayPic))
figure,montage({grayPic,laplacianFilter(grayPic),laplacianFilter2(grayPic)})

%Extra Filter

%extra = lowPassfilter(size(grayPic),pi/2);




%%
%Iris try again
picToRead = './iris.bmp';
pic = im2double(imread(picToRead));

%figure,imshow(pic)
[~, ~, numberOfColorChannels] = size(pic);
if numberOfColorChannels > 1
    % It's a true color RGB image.  We need to convert to gray scale.
    grayPic = rgb2gray(pic);
else
    % It's already gray scale.  No need to convert.
    grayPic = pic;
    fprintf("already gray")
end

%histogram Equilization to get an idea of where to threshold.
[J,T] = histeq(grayPic);
figure, plot((0:255)/255,T);
figure, imhist(T,64)

% threshold 
t = 0.16;
%thresholding the gray image.
binaryImage = grayPic > t;
[row,col] = size(binaryImage);

edgePic = sobFilter(binaryImage);
meanPic = medFilt(edgePic);

ECCENTRICITY_TH = 0.5;
% look for region properties
s = regionprops(binaryImage, grayPic, {'Eccentricity', 'Centroid'});
% keep and plot those close to be a "circle"
numRegions = numel(s);
for k = 1:numRegions
  if s(k).Eccentricity < ECCENTRICITY_TH
    px = s(k).Centroid(1);
    py = s(k).Centroid(2);
    ptxt = ['  \leftarrow(' num2str(px) ', ' num2str(py) ')'];
    %center is (px,py) NOTE: px changes form 179 to 177 depending on
    %threshold being .16 or .17.g,  So that is extremly interesting.
    text(px, py, ptxt, 'Background', [.5 .5 .5], 'Color', 'blue');
    plot(px, py, 'wd', 'MarkerFaceColor', 'blue');
  end
end
newPic1 = circle(px,py,35);
newPic2 = circle(px,py,45);
centers = [px,py];

[row,col] = size(grayPic);
for x=1:row
    for y=2:col
        if round(sqrt( (x-(px-43))^2 + (y-(py+13))^2 )) <=45
            meanPic(x,y) = 255;
        end
    end
end
for x=1:row
    for y=2:col
        if round(sqrt( (x-(px-43))^2 + (y-(py+13))^2 )) <=35
            meanPic(x,y) = 0;
        end
    end
end

figure,montage({pic,grayPic,binaryImage,edgePic,meanPic})
%imsave(montage({pic,grayPic,binaryImage,edgePic,meanPic}))
%% All Functions

%Function to find the average pic of any picture given. 
function avgPicOut = avgFilter(usePic)
    avgFilt=[1 1 1;
             1 1 1;
             1 1 1];%Divide matrix by 9 for the average of them all
    %xDone=3;
    %for x=0:xDone%Should I be looping this to get a better smoothing?
        avgFilt = avgFilt * (1/9);
    
        %avgFilt=[2 2 2;2 2 2;2 2 2]/9;
        %avgPicOut = filter2(usePic,avgFilt,'full');
        %mesh(avgPicOut)
        avgPicOut = conv2(usePic,avgFilt);
    %end
end

function med = medFilt(usePic)%There has to be a better way to implement this...
    [rows, col] = size(usePic);%size of grayscale image
    picSalt = usePic;%Unsure why I did this, to lazy to change now. 
    pad=zeros(rows,col);
    for i=2:rows-1
        for j=2:col-1
            %Make 3x3 mask
            filter = [picSalt(i-1,j-1),picSalt(i-1,j),picSalt(i-1,j+1),picSalt(i,j-1),picSalt(i,j),picSalt(i,j+1),picSalt(i+1,j-1),picSalt(i+1,j),picSalt(i+1,j+1)];
            pad(i,j)= median(filter);%function that just return median value
        end
    end
    med = pad;
end

function sobelKernel = sobFilter(usePic)
    sobFiltOne=[-1 0 +1;
                -2 0 +2;
                -1 0 +1];%xDirection is done
    sobFiltTwo=[+1 +2 +1;
                 0 0 0;
                -1 -2 -1];%yDirection is done
    %sobelFiltOneImage = imfilter(usePic,sobFiltOne,'same');
    %sobelFiltTwoImage = imfilter(usePic,sobFiltTwo,'same');
        %Working is above.
    sobelFiltOneImage = conv2(usePic,sobFiltOne);%Another function to loop through picture to apply Kernel
    sobelFiltTwoImage = conv2(usePic,sobFiltTwo);
    %sobelFilt#Image = conv2(usePic,sobfilt#,'full');
        %Why does filter2 flip the image upside down?
            %conv2(pic, mask) = filter2(rot90(mask,2), pic)
            %conv2 is a bit faster than fitler, no reason for me to use
            %filter2 here.  
    
    sobelKernel = sqrt(sobelFiltOneImage.^2 + sobelFiltTwoImage.^2);
      %Edges detected better due to reading the image as a double from the
      %begging. 
end

function lapLacian = laplacianFilter(usePic)
    laplFilt=[-1 -1 -1; 
              -1 8 -1; 
              -1 -1 -1];
    %laplacianFilt = filter2(usePic,laplFilt,'self');
    lapLacianFilt = conv2(usePic,laplFilt,'same');%This was lapLacianFilt
    
    lapLacian = usePic + lapLacianFilt;%Sharpen the image.
end

function lapLacian = laplacianFilter2(usePic)
    laplFilt=[-1 -1 -1; 
              -1 8 -1; 
              -1 -1 -1];
    %laplacianFilt = filter2(usePic,laplFilt,'self');
    lapLacian = conv2(usePic,laplFilt,'same');%This was lapLacianFilt
   
end

function h = circle(x,y,r)
    hold on
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit);
    hold off
end



















