%{
Referenece:
https://www.itread01.com/content/1542714604.html

%}

img=imread('book-cover-blurred.tif');
img=im2double(img);

len=28;
theta=14;
PSF=fspecial("motion",len,theta);
blurred=imfilter(img,PSF,'circular');
% motion blur
subplot(2,3,1),imshow(img),title("原圖")
subplot(2,3,2),imshow(blurred),title("運動模糊")
If=fft2(blurred);
Pf=fft2(PSF,500,500);
deblurred=ifft2(If./Pf);
subplot(2,3,3),imshow(deblurred),title("直接逆濾波")

mean=0;
var=0.0001;
noised=imnoise(blurred,'gaussian',mean,var);
subplot(2,3,4),imshow(noised),title("模糊加噪聲")

Ifn=fft2(noised);
deblurredn=ifft2(Ifn./Pf);
subplot(2,3,5),imshow(deblurredn),title("直接逆濾波")