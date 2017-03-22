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

% OPTIONAL 2 ------------------------------------------------------------
% Improve the saving by removing the incomplete pixels or adding neural
% gray to incomplete pixels.


% OPTIONAL 3 ------------------------------------------------------------
% Sharpening


% OPTIONAL 4 ------------------------------------------------------------
% Remove image defects using morphology techniques








