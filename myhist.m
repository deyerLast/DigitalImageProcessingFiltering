%this program is to get the normalized histogram of the image

function [his]=myhist(x)
x=floor(x);
r=size(x,1);
c=size(x,2);
his=zeros(1,256);
%calculate the histogram
for i=1:r
   for j=1:c
      his(x(i,j)+1)=his(x(i,j)+1)+1;
   end
end
his=his./c./r;