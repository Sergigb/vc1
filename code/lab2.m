%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%       xxxxxxxxxxxxxxxxxxxxxxxx       %%%         LAB x         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Students Name and NIU:                                        %%%
%%%  Surename(s), Name (NIU)                                       %%%
%%%  Email:                                                        %%%
%%%  Surename(s), Name (NIU)                                       %%%
%%%  Email:                                                        %%%
%%%  Matlab / Octave:                                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Deliverables until We. Mar. 30th. 23:00h                       %%%
%%% xxxxx							                               %%%
%%% + xxxxxxx                                                      %%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Hello! Welcome to the x. 

% TODO. Please, copy the images folder in the parent directory to make the 
% code work properly.
addpath('../img');
addpath('../images');

% Please don't remove that, if you are using octave just comment it!
clearvars,
close all,
clc,

% Image names
file_names = {'00149v.jpg', '00153v.jpg', '00194v.jpg', '00458v.jpg', 
              '01167v.jpg', '00163v.jpg', '00398v.jpg', '00600v.jpg'};  

% TODO. Load the first of the images to work with:
im1 = imread(file_names{3});

% TODO. Load the second of the images to work with:
im2 = imread(file_names{8});

% Problem 1. Crop the images using the simple crop (+1) ------------------

% TODO. Compute the image size of both images
[height1, width1] = size(im1);
[height2, width2] = size(im2);

% TODO. Crop the images in 3 parts using the measurements computed above.
% Take into accountthat they are not in the RGB order. Check if they are
% in RGB, BRG, BGR, etc.
blue_channel1= im1(int32(1:height1/3),:);
green_channel1 = im1(int32(height1/3:height1*(2/3)-1),:);
red_channel1 = im1(int32(height1*(2/3):height1-1),:);

red_channel2   = im2(int32(1:height2/3),:);
green_channel2 = im2(int32(height2/3:height2*(2/3)-1),:);
blue_channel2 = im2(int32(height2*(2/3):height2-1),:);

% Ploting the three channels in a 1x3 subplot
figure(1),

iptsetpref('ImshowAxesVisible','on');

subplot(1,3,1);
imshow(red_channel1);
title('red channel');
subplot(1,3,2);
imshow(green_channel1);
title('green channel');
subplot(1,3,3);
imshow(blue_channel1);
title('blue channel');

figure(2),

subplot(1,3,1);
imshow(red_channel2);
title('red channel');
subplot(1,3,2);
imshow(green_channel2);
title('green channel');
subplot(1,3,3);
imshow(blue_channel2);
title('blue channel');

% Problem 2. Correlations -----------------------------------------------
% Here we will crop just the central part of the image in order to achieve
% a better correlation.

sz=size(red_channel1);
rangef=round(sz(1)/4):round(3*sz(1)/4);
rangec=round(sz(2)/4):round(3*sz(2)/4);

red_cut1 = red_channel1(rangef,rangec);     
green_cut1 =  green_channel1(rangef,rangec);
blue_cut1 =  blue_channel1(rangef,rangec);

sz=size(red_channel2);
rangef=round(sz(1)/4):round(3*sz(1)/4);
rangec=round(sz(2)/4):round(3*sz(2)/4);

red_cut2 = red_channel2(rangef,rangec);     
green_cut2 =  green_channel2(rangef,rangec);
blue_cut2 =  blue_channel2(rangef,rangec);

% Spatial correlation (+1.5) ----------------------------------------------
% help conv2

% TODO. Select one channel as reference and correlate the other two using
% Spatial Correlation.

reference1 = double(green_cut1);
reference2 = double(blue_cut2);

% TODO. Here you can do the magic trick explained on the slides to achive a 
% good correlation

% TODO. Compute the spatial correlation of the reference channel with other
% channels for the two images.

Green_Red1 = conv2((double(red_cut1-mean2(reference1))),flipud(reference1-mean2(reference1)), 'same');
Green_Blue1 = conv2((double(blue_cut1-mean2(reference1))), flipud(reference1-mean2(reference1)), 'same');

Blue_Red2 = conv2((double((red_cut2-mean2(reference2)))),flipud(reference2-mean2(reference2)), 'same');
Blue_Green2 = conv2((double((green_cut2-mean2(reference2)))), flipud(reference2-mean2(reference2)), 'same');

% TODO. Find the coordinates of the correlation correlation 
% help find
[value1_red, location1_red] = max(Green_Red1(:));
[R1_red,C1_red] = ind2sub(size(Green_Red1),location1_red); 

[value1_blue, location1_blue] = max(Green_Blue1(:));
[R1_blue,C1_blue] = ind2sub(size(Green_Blue1),location1_blue);

[value2_red, location2_red] = max(Blue_Red2(:));
[R2_red,C2_red] = ind2sub(size(Blue_Red2),location2_red); 

[value2_green, location2_green] = max(Blue_Green2(:));
[R2_green,C2_green] = ind2sub(size(Blue_Green2),location2_green);

[h_temp2, w_temp2] = size(Blue_Red2);
h_temp2 = int32(h_temp2/2);
w_temp2 = int32(w_temp2/2);


% TODO. Shifting the position in the correlated channel
% help circshift
red_shifted_1 = circshift(red_channel1, [h_temp2 - R1_red, w_temp2-C1_red]);
blue_shifted_1 = circshift(blue_channel1, [h_temp2-R1_blue/2, w_temp2-C1_blue]);

red_shifted_2 = circshift(red_channel2, [h_temp2 - R2_red, w_temp2-C2_red]);
green_shifted_2 = circshift(green_channel2, [h_temp2-R2_green, w_temp2-C2_green]);

% TODO. Mix che shifted channels and the reference channel in a single 
% correlated image.
% help cat
spatial_correlated1 = cat(3, red_shifted_1, green_channel1, blue_shifted_1);
spatial_correlated2 = cat(3, blue_channel2, green_shifted_2, red_shifted_2);

% TODO. Show the results (just uncomment)
figure(3),
subplot(1,2,1), imshow(spatial_correlated1);
subplot(1,2,2), imshow(spatial_correlated2);
title('spatial\_correlated');

% Normalized cross correlation (+1.5) -------------------------------------
% help normxcorr2

% TODO. Correlate the reference channel with one of the two other.
Green_Red1_cross = normxcorr2(reference1, double((red_channel1)));
Green_Blue1_cross = normxcorr2(reference1, double((blue_channel1)));

Blue_Red2_cross = normxcorr2(reference2, double((red_channel2)));
Blue_Green2_cross = normxcorr2(reference2, double((green_channel2)));

% TODO. Find the coordinates of the max of R (the correlation maximum)
% help find
[value1_red_cross, location1_red_cross] = max(Green_Red1_cross(:));             %image 1
[row_green_red1,col_green_red1] = ind2sub(size(Green_Red1_cross),location1_red_cross); 

[value1_blue_cross, location1_blue_cross] = max(Green_Blue1_cross(:));
[row_green_blue1,col_green_blue1] = ind2sub(size(Green_Blue1_cross),location1_blue_cross); 


[value2_red_cross, location2_red_cross] = max(Blue_Red2_cross(:));             %image 2
[row_blue_red2,col_blue_red2] = ind2sub(size(Blue_Red2_cross),location2_red_cross); 

[value2_green_cross, location2_green_cross] = max(Blue_Green2_cross(:));
[row_blue_green2,col_blue_green2] = ind2sub(size(Blue_Green2_cross),location2_green_cross); 

% TODO. Shifting the position in the correlated channel
% help circshift
[h,w] = size(Green_Red1_cross);
[h2,w2] = size(Blue_Red2_cross);

red_shifted_1_cross = circshift(red_channel1, int32([h/2 - row_green_red1, w/2 - col_green_red1]));
blue_shifted_1_cross = circshift(blue_channel1, int32([h/2 - row_green_blue1, w/2 - col_green_blue1]));

red_shifted_2_cross = circshift(red_channel2, int32([h2/2 - row_blue_red2, w2/2 - col_blue_red2]));
green_shifted_2_cross = circshift(green_channel2, int32([h2/2 - row_blue_green2, w2/2 - col_blue_green2]));

% TODO. Mix che shifted channels and the reference channel in a single 
% correlated image.
% help cat
spatial_cross_correlated1 = cat(3, red_shifted_1_cross, green_channel1, blue_shifted_1_cross);
spatial_cross_correlated2 = cat(3, blue_channel2, green_shifted_2_cross, red_shifted_2_cross ); %los canales rojo y azul estan intercambiados

% % TODO. Show the results (just uncomment)
figure(5),
subplot(1,2,1), imshow(spatial_cross_correlated1);
subplot(1,2,2), imshow(spatial_cross_correlated2);
title('normalxcorrelated');

% Fourier domain correlation (+2) ----------------------------------------
% help fft2
% help ifft2


% Fourier domain correlations:
fft_reference1 = fft2(reference1);                          %domain correlation image 1
fft_red1 = fft_reference1.*conj(fft2(red_cut1));            %domain correlation red
domain_correlation_red1 = ifft2(fft_red1);

fft_blue1 = fft_reference1.*conj(fft2(blue_cut1));          %domain correlation blue
domain_correlation_blue1 = ifft2(fft_blue1);

fft_reference2 = fft2(reference2);                          %domain correlation image 1
fft_red2 = fft_reference2.*conj(fft2(red_cut2));            %domain correlation red
domain_correlation_red2 = ifft2(fft_red2);

fft_green2 = fft_reference2.*conj(fft2(green_cut2));        %domain correlation green
domain_correlation_green2 = ifft2(fft_green2);

%Coordinates where the correlation is maximum:
[value1_red_dom, location1_red_dom] = max(domain_correlation_red1(:));             %shifts image 1
[row_green_red1_dom,col_green_red1_dom] = ind2sub(size(domain_correlation_red1),location1_red_dom); 

[value1_blue_dom, location1_blue_dom] = max(domain_correlation_blue1(:));
[row_green_blue1_dom,col_green_blue1_dom] = ind2sub(size(domain_correlation_blue1),location1_blue_dom); 

[value2_red_dom, location2_red_dom] = max(domain_correlation_red2(:));             %shifts image 2
[row_blue_red2_dom,col_blue_red2_dom] = ind2sub(size(domain_correlation_red2),location2_red_dom); 

[value2_green_dom, location2_green_dom] = max(domain_correlation_green2(:));
[row_blue_green2_dom,col_blue_green2_dom] = ind2sub(size(domain_correlation_green2),location2_green_dom); 

%Image shifts:
[h_do,w_do] = size(domain_correlation_red1);
[h_do2,w_do2] = size(domain_correlation_red2);

red_shifted_1_dom = circshift(red_channel1, [row_green_red1_dom, col_green_red1_dom]);   %circshift image 1
blue_shifted_1_dom = circshift(blue_channel1, [-(h_do - row_green_blue1_dom),-(w_do - col_green_blue1_dom)]);

red_shifted_2_dom = circshift(red_channel2, [-(h_do2 - row_blue_red2_dom), -(w_do2 - col_blue_red2_dom)]);   %circshift image 2
green_shifted_2_dom = circshift(green_channel2, [-(h_do2 - row_blue_green2_dom), -(w_do2 - col_blue_green2_dom)]);

%concatenation
fourier_correlated1 = cat(3, red_shifted_1_dom, green_channel1, blue_shifted_1_dom);
fourier_correlated2 = cat(3, blue_channel2, green_shifted_2_dom, red_shifted_2_dom);

figure(4),
subplot(1,2,1), imshow(fourier_correlated1);
subplot(1,2,2), imshow(fourier_correlated2);
title('Fourier Domain');

% Fourier phase correlation (+2) -----------------------------------------
% help fft2
% help ifft2

% phase correlation in fourier domain:
fft_reference1 = fft2(reference1);                          %phase correlation image 1
cps_red1 = fft_reference1.*conj(fft2(red_cut1));            %phase correlation red channel
phase_correlation_red1 = ifft2(cps_red1./abs(cps_red1));

cps_blue1 = fft_reference1.*conj(fft2(blue_cut1));          %phase correlation blue channel
phase_correlation_blue1 = ifft2(cps_blue1./abs(cps_blue1));

fft_reference2 = fft2(reference2);                          %phase correlation image 2
cps_red2 = fft_reference2.*conj(fft2(red_cut2));            %phase correlation red channel
phase_correlation_red2 = ifft2(cps_red2./abs(cps_red2));

cps_green2 = fft_reference2.*conj(fft2(green_cut2));        %phase correlation green channel
phase_correlation_green2 = ifft2(cps_green2./abs(cps_green2));

%Coordinates where the correlation is maximum:
[value1_red_four, location1_red_four] = max(phase_correlation_red1(:));             %shifts image 1
[row_green_red1_four,col_green_red1_four] = ind2sub(size(phase_correlation_red1),location1_red_four); 

[value1_blue_four, location1_blue_four] = max(phase_correlation_blue1(:));
[row_green_blue1_four,col_green_blue1_four] = ind2sub(size(phase_correlation_blue1),location1_blue_four); 

[value2_red_four, location2_red_four] = max(phase_correlation_red2(:));             %shifts image 2
[row_blue_red2_four,col_blue_red2_four] = ind2sub(size(phase_correlation_red2),location2_red_four); 

[value2_green_four, location2_green_four] = max(phase_correlation_green2(:));
[row_blue_green2_four,col_blue_green2_four] = ind2sub(size(phase_correlation_green2),location2_green_four); 

%Image shifts:
[h_ph,w_ph] = size(phase_correlation_red2);

red_shifted_1_four = circshift(red_channel1, [row_green_red1_four, col_green_red1_four]);   %circshift image 1
blue_shifted_1_four = circshift(blue_channel1, [-(h_ph - row_green_blue1_four),-(w_ph - col_green_blue1_four)]);

[h_ph2,w_ph2] = size(phase_correlation_red2);
red_shifted_2_four = circshift(red_channel2, [-(h_ph2 - row_blue_red2_four), -(w_ph2 - col_blue_red2_four)]);   %circshift image 2
green_shifted_2_four = circshift(green_channel2, [-(h_ph2 - row_blue_green2_four), -(w_ph2 - col_blue_green2_four)]);

%concatenation
phase_correlated1 = (cat(3, red_shifted_1_four, green_channel1, blue_shifted_1_four));
phase_correlated2 = (cat(3, blue_channel2, green_shifted_2_four, red_shifted_2_four));

figure(6),
subplot(1,2,1), imshow(phase_correlated1);
subplot(1,2,2), imshow(phase_correlated2);
title('Fourier Phase');



%% Save the images ------------------------------------------------------

% TODO. Save the images
% name_mode.jpg where name is the previous image name and mode is the
% correlation you have applied.

if ~exist('../results', 'dir')
  mkdir('../results');
end

imwrite(spatial_correlated1,'../results/space_correlation1.png');
%imwrite(spatial_correlated2,'../results/space_correlation2.png');
imwrite(spatial_cross_correlated1,'../results/spatial_cross_correlated1.png');
imwrite(spatial_cross_correlated2,'../results/spatial_cross_correlated2.png');
imwrite(fourier_correlated1,'../results/fourier_correlated1.png');
imwrite(fourier_correlated2,'../results/fourier_correlated2.png');
imwrite(phase_correlated1, '../results/phase_correlated1.png');
imwrite(phase_correlated2, '../results/phase_correlated2.png');


% OPTIONAL 1 ------------------------------------------------------------
% Improve the crop

supercell = {red_channel1, green_channel1, blue_channel1, red_channel2, green_channel2, blue_channel2};
bbxy = zeros(6,2);      %6x2 matrix where we store the (upper left) coordinates of the different bounding boxes (3 channels per image)
bbwh = zeros(6,2);      %6x2 matrix where we store the width and the height of the different bounding boxes

for i=1:6
    channel = cell2mat(supercell(i));

    [w,h] = size(channel);
    filt = zeros(w,h);
    w2=w;
    h2=h;

    filt(:,:) = channel(:,:)<15;        %threshold to eliminate the white borders of the images
    st = regionprops(filt, 'BoundingBox' );
    thisBB1 = st(1).BoundingBox;        %creates a bounding box around the part of the image we want to crop, thus eliminating the white contours
    crop_image = channel(int32(thisBB1(2):thisBB1(4)), int32(thisBB1(1):thisBB1(3)));

    [w,h] = size(crop_image);
    filt=zeros(w,h);
    filt(:,:) = not(crop_image(:,:)<15);    %threshold to eliminate the black borders of the images

    se = strel('line',9,70);        %removing the black contours of the images is more difficult because they are less uniform, 
    filt = imerode(filt, se);       %if we only use a threshold, the bounding box cannot fully eliminate the black borders
    se = strel('line',10,1);        %these morphological operations help to enhance the filter and create the bounding box
    filt = imerode(filt, se);
    se = strel('disk',5);
    filt = imerode(filt, se);

    st = regionprops(filt, 'BoundingBox' );
    thisBB2 = st(1).BoundingBox;
    
    bbxy(i,:) = [thisBB2(1)+thisBB1(1), thisBB2(2)+thisBB1(2)];     %we store the coordinates and the sizes of all bounding boxes 
    bbwh(i,:) = [thisBB2(3)+(w2-thisBB1(3)), thisBB2(4)+(h2-thisBB1(4))];   %of the different image channels
end

maxxy1 = int32(max(bbxy(1:3,:)));   %to ensure that the 3 channels of each image have the same size, we use the smallest bounding
maxxy2 = int32(max(bbxy(4:6,:)));   %boxes we can create with the different parameters??
minwh1 = int32(min(bbwh(1:3,:)));   %if we use the smallest bounding boxes possible, we will ensure that the cropping effectively
minwh2 = int32(min(bbwh(4:6,:)));   %eliminates the borders in all the different channels of the image.

figure(666),
subplot(2,3,1), imshow(red_channel1(maxxy1(1):minwh1(1), maxxy1(2):minwh1(2)));
subplot(2,3,2), imshow(green_channel1(maxxy1(1):minwh1(1), maxxy1(2):minwh1(2)));
subplot(2,3,3), imshow(blue_channel1(maxxy1(1):minwh1(1), maxxy1(2):minwh1(2)));

subplot(2,3,4), imshow(red_channel2(maxxy2(1):minwh2(1), maxxy2(2):minwh1(2)));
subplot(2,3,5), imshow(green_channel2(maxxy2(1):minwh2(1), maxxy2(2):minwh1(2)));
subplot(2,3,6), imshow(blue_channel2(maxxy2(1):minwh2(1), maxxy2(2):minwh1(2)));


% OPTIONAL 2 ------------------------------------------------------------
% Improve the saving by removing the incomplete pixels or adding neural
% gray to incomplete pixels.

% OPTIONAL 3 ------------------------------------------------------------
% Sharpening
image1_ycbcr = rgb2ycbcr(spatial_cross_correlated1);    %conversion from rgb to ycbcr
image1_ycbcr(:, :, 1) = filter2(fspecial('unsharp'), double(image1_ycbcr(:, :, 1)));    %to sharpen the image, we use a high-pass filter in the luma dimension of the image
image1_sharp = ycbcr2rgb(image1_ycbcr);


image2_ycbcr = rgb2ycbcr(spatial_cross_correlated2);
image2_ycbcr(:, :, 1) = filter2(fspecial('unsharp'), double(image2_ycbcr(:, :, 1)));
image2_sharp = ycbcr2rgb(image2_ycbcr);

figure(9),
subplot(1,2,1), imshow(image1_sharp);
subplot(1,2,2), imshow(image2_sharp);
title('Sharpening');


% OPTIONAL 4 ------------------------------------------------------------
% Remove image defects using morphology techniques




