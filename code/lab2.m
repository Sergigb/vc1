%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%       COMPUTER VISION LAB 2017       %%%         LAB 2         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Students Name and NIU:                                        %%%
%%%  Surename(s), Name (NIU)                                       %%%
%%%  Email:                                                        %%%
%%%  Surename(s), Name (NIU)                                       %%%
%%%  Email:                                                        %%%
%%%  Matlab / Octave:                                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Deliverables until We. Mar. 30th. 23:00h                       %%%
%%% CERBERO							                               %%%
%%% + lab2.m                                                       %%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Hello! Welcome to the computer vision LAB 2. 

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

reference1=double(green_cut1);
reference2=double(blue_cut2);

% TODO. Here you can do the magic trick explained on the slides to achive a 
% good correlation

% TODO. Compute the spatial correlation of the reference channel with other
% channels for the two images.

Green_Red1 = conv2((flipud(double(red_channel1))),(reference1), 'same');
Green_Blue1 = conv2((flipud(double(blue_channel1))), (reference1), 'same');

%Blue_Red2 = etc
%Blue_Green2= etc

% TODO. Find the coordinates of the correlation correlation 
% help find

[value1_red, location1_red] = max(Green_Red1(:));
[R1_red,C1_red] = ind2sub(size(Green_Red1),location1_red); 

[value1_blue, location1_blue] = max(Green_Blue1(:));
[R1_blue,C1_blue] = ind2sub(size(Green_Blue1),location1_blue);


[h_temp, w_temp] = size(Green_Red1);
h_temp = int32(h_temp/4);
w_temp = int32(w_temp/4);


%Green_Red1(h_temp ,w_temp) = 0.6;
%Green_Blue1(h_temp ,w_temp) = 0.6;
% 
% Green_Red1(R1_red,C1_red) = 0.6;
% Green_Blue1(R1_blue,C1_blue) = 0.6;
% 
% figure(14),
% imshow((Green_Blue1), [])
% title('blue');
% figure(15),
% imshow((Green_Red1), [])
% title('red');

% mc_red_greem_2 = 
% mc_red_blue_2 = 

% TODO. Shifting the position in the correlated channel
% help circshift

red_shifted_1 = circshift(red_channel1, [-1*R1_red-h_temp, -1*C1_red-w_temp]);
blue_shifted_1 = circshift(blue_channel1, [R1_blue+h_temp, -1*C1_blue+w_temp]);

% TODO. Mix che shifted channels and the reference channel in a single 
% correlated image.
% help cat
 spatial_correlated1 = cat(3, red_shifted_1, green_channel1, blue_shifted_1);
% spatial_correlated2 = 

% TODO. Show the results (just uncomment)
figure(3),
subplot(1,1,1), imshow(spatial_correlated1);
%subplot(1,2,2), imshow(spatial_correlated2);
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



% Fourier phase correlation (+2) -----------------------------------------
% help fft2
% help ifft2



%% Save the images ------------------------------------------------------

% TODO. Save the images
% name_mode.jpg where name is the previous image name and mode is the
% correlation you have applied.

% OPTIONAL 2 ------------------------------------------------------------
% Improve the crop

% OPTIONAL 2 ------------------------------------------------------------
% Improve the saving by removing the incomplete pixels or adding neural
% gray to incomplete pixels.


% OPTIONAL 3 ------------------------------------------------------------
% Sharpening


% OPTIONAL 4 ------------------------------------------------------------
% Remove image defects using morphology techniques








