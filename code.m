%Training
K_matrices=zeros(512,512,12);
for i=1:12
    img=im2double(imread(strcat(int2str(i),'[1]','.gif')));
    n=imnoise(img,'gaussian',0,0.001)-img;
    N=fft2(n);
    F=fft2(img);
    K_matrices(:,:,i)=(abs(N).^2)./(abs(F).^2);
end
K_final=zeros(512,512);
for i=1:12
    K_final=K_final+K_matrices(:,:,i);
end
K_final=K_final/12;
%%
%Reading the image in Grayscale
originalImage=im2double(imread('test.gif'));
[m,n]=size(originalImage);

%Add Gaussian Noise
h=fspecial('gaussian',[5 5],5);
im_gauss=imfilter(originalImage,h,'conv','symmetric');

%Blur the iamge
mean = 0;
variance = 0.01;
im_blur = imnoise(im_gauss,'gaussian',mean,variance);

%FFT of Error function
H=fft2(h,m,n);
H_func=abs(H)/max(max(abs(H)));

%FFT of Distorted[Blurred] image
G=fft2(im_blur);
G_func=abs(G)*255/max(max(abs(G)));

%FFT of Grayscale Original Image
F=fft2(originalImage);
F_func=abs(F)*255/max(max(abs(F)));



%Restoration of Image
H_func = conj(H);
fraction = H_func./((abs(H).^2)+K_final);
temp = G.*fraction;
temp2=ifft2(temp);
restoredImage=abs(temp2)/max(max(abs(temp2)));

%Viewing of images
subplot(1,3,1)
imshow(originalImage)
title("GrayScale Image")
subplot(1,3,2)
imshow(im_blur)
title("Blurred Image")
subplot(1,3,3)
imshow(restoredImage)
title("Restored Image")
%%
%Metric - PSNR, MSE
psnrval = psnr(restoredImage,originalImage);
disp("PSNR")
disp(psnrval)
%%
mseval = immse(restoredImage,originalImage);
disp("Mean Square Error")
disp(mseval)