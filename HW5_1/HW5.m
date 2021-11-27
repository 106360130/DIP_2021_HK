clear all;
close all;


%{
images = load('Fig5.25.jpg');   % Load the mat file containing images
%m_blur = images.mandrill_blurred;  % Extract the first image 
figure;imshow(images)
%imagesc(m_blur);                   % display the blurred image
h = fspecial('gaussian',[25 25],15);   % 25x25 Gaussian blur function with sigma = 15
hf = fft2(h,size(m_blur,1),size(m_blur,2));      
m_deblur = real(ifft2(m_blur)./hf);      %inverse filter 
figure(2)
%imagesc(m_deblur)   % Display deblurred image
%}

                     

clc;
clear all;
close all;

%讀取影像
f=rgb2gray(im2double(imread('Fig5.25.jpg')));
f=imresize(f,[256 256])
figure,(imshow(f))
title("f");
[M,N]=size(f);
%讀取影像


% k=2.5;
%  for i=1:size(f,1)
%      for j=1:size(f,2)
%          h(i,j)=exp((-k)*((i-M/2)^2+(j-N/2)^2)^(5/6));
%       end
%  end

%製造高斯模糊
h=fspecial('gaussian',260,2);
g=(imfilter(f,h,'circular'));  %再次高斯模糊後的影像
figure,imshow(g,[]);
title("g")
%製造高斯模糊


%將高斯模糊後的像做FFT
G = fftshift(fft2(g));
figure,imshow(log(abs(G)),[]);
title("log(abs(G))")
H = fftshift(fft2(h));
figure,imshow(log(abs(H)),[]);
title("log(abs(H))")
%將高斯模糊後的像做FFT


F = zeros(size(f));
R=70;
for u=1:size(f,2)
    for v=1:size(f,1)
        du = u - size(f,2)/2;
        dv = v - size(f,1)/2;
        if du^2 + dv^2 <= R^2;
        F(v,u) = G(v,u)./H(v,u);
        end
    end
end


figure,imshow(log(abs(F)),[]);
title("log(abs(F))")


test = ifftshift(F);
figure,imshow(log(abs(test)),[]);
fRestored = abs(ifft2(test));
figure,imshow(fRestored, []);
title("fRestored")
