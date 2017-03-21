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


%Blue_Red2 = etc
%Blue_Green2= etc

% TODO. Find the coordinates of the correlation correlation 
% help find


% TODO. Shifting the position in the correlated channel
% help circshift

% green_shifted_2 = 
% blue_shifted_2 = 

% TODO. Mix che shifted channels and the reference channel in a single 
% correlated image.
% help cat

% TODO. Show the results (just uncomment)
% figure(3),
% subplot(1,1,1), imshow(spatial_correlated1);
% subplot(1,2,2), imshow(spatial_correlated2);
% title('spatial\_correlated');

% Normalized cross correlation (+1.5) -------------------------------------
% help normxcorr2

% TODO. Correlate the reference channel with one of the two other.

% TODO. Find the coordinates of the max of R (the correlation maximum)
% help find

% TODO. Shifting the position in the correlated channel
% help circshift

% TODO. Mix che shifted channels and the reference channel in a single 
% correlated image.
% help cat

% % TODO. Show the results (just uncomment)

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








