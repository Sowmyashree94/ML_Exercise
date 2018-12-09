clc
clear

%to select p elements
startIndex = 1;
p = 160;

%image to be reconstructed and compared
imageIndex = 12;

%% a) Write a routine to load all images into a big data matrix
% X is the big data matrix
%get the list of all files in yalefaces directory
list =  dir('yalefaces');
X = [];

for index = 1:length(list)
    fileNames = list(index).name;
    if 1==contains(fileNames,"subject")
       read = imread(strcat('yalefaces/', fileNames)); 
       %convert each image to 1-d array and append
       reshaped = reshape(read,1,[]);
       X = vertcat(X,reshaped);
    end
end

[m,n] = size(X);

%% b) Compute the mean face and center the whole data matrix
mu=sum(X,1)/n;
%after data centering
Xtilde = double(X) - (ones(m,1)*mu);

%% c) Compute the singular value decomposition for the centered data matrix 
[U,S,V] = svd(Xtilde, "econ");

%% d) Find the p-dimensional representations
% S matrix has the eigenvectors
% vp = V(:,[startIndex:p]);
% Z = Xtilde*vp;

%% e) Reconstruct the image
% Xdash = (ones(m,1)*mu) +(Z*vp');
% img_construct1d = Xdash(imageIndex,:);
% img_construct = reshape(img_construct1d,243,320);
% 
% img_original1d = X(imageIndex,:);
% img_original = reshape(img_original1d,243,320);
% 
% subplot(1,2,1), imagesc(img_original)
% title(['Original image: ',num2str(list(imageIndex).name),'  p: ',num2str(p)]);
% subplot(1,2,2), imagesc(img_construct)
% title(['Reconstructed image: ',num2str(list(imageIndex).name),'  p: ',num2str(p)]);

% %error for a single image
% error = sum((double(img_original1d) - img_construct1d).^2)
% %error for the entire dataset for a particular p
% error1 = sum(sum((double(X) - Xdash).^2),2)


count =1;
img_original1d = X(imageIndex,:);
img_original = reshape(img_original1d,243,320);
subplot(2,3,1), imagesc(img_original)
title(['Original image: ',num2str(list(imageIndex).name),'  p: ',num2str(p)]);
for p = [5,10,50,100,150]
    vp = V(:,[startIndex:p]);
    Z = Xtilde*vp;
    Xdash = (ones(m,1)*mu) +(Z*vp');
    img_construct1d = Xdash(imageIndex,:);
    img_construct = reshape(img_construct1d,243,320);
    count = count+1;
    subplot(2,3,count), imagesc(img_construct)
    title(['Reconstructed image: ',num2str(list(imageIndex).name),'  p: ',num2str(p)]);
    error1 = sum(sum((double(X) - Xdash).^2),2)
end

%covariance matrix
% C_original = cov(X);
