clear all
clc
list =  dir('yalefaces_cropBackground');
X = [];

for index = 1:length(list)
    fileNames = list(index).name;
    if 1==contains(fileNames,"subject")
       read = imread(strcat('yalefaces_cropBackground/', fileNames)); 
       %convert each image to 1-d array and append
       reshaped = reshape(read,1,[]);
       X = vertcat(X,reshaped);
    end
end

%number of clusters
k = 4;
% figure
% hold on

[m,n] = size(X);

%%a
%idx = rnk 
% C = centroid
%sumd = min of the euclidean error
% replicates = number of times the kmeans has to run by choosing different
% values of initial means
% maxIter = number of iteration in each replicate
opts = statset('Display','final');
[idx, C, sumd] = kmeans(double(X),k,'Distance','sqeuclidean','Replicates',10,'Options',opts);

%plot each cluster data
figure(1)
title("Center faces");

for i = 1:k
    cluster = reshape(C(i,:),[243,160]);
    subplot(2,4,i),imagesc (cluster); title (['Cluster ',num2str(i)]);
end

% plot samples for each cluster
sample = 1;
figure(1)
for i = 1:k
    img_sample = X(idx==i,:);
    img_sample_disp = reshape(img_sample(sample,:),243,160);
    subplot(2,4,4+i),imagesc(img_sample_disp),title(['Sample image for cluster',num2str(i)]);
end

%%b
cluster_error = [];
count = 1;
for i = 4 : 15
    [idx, C, sumd] = kmeans(double(X),i,'Distance','sqeuclidean','Replicates',10,'Options',opts);
    cluster_error(count) = sum(sumd);
    count = count+1;
end

figure(2)
subplot(2,1,1),plot(4:15,cluster_error)
xlabel("number of clusters");
ylabel("clustering error");
title("Plot of clustering error over number of clusters");


%%c 
p=20;
mu=sum(X,1)/n;
Xtilde = double(X) - (ones(m,1)*mu);
[U,S,V] = svd(Xtilde, "econ");
vp = V(:,[1:p]);
Z = Xtilde*vp;
% Xdash is pca data
Xdash = (ones(m,1)*mu) +(Z*vp');
cluster_error_pca = [];
count = 1;
for i = 4 : 15
    [idx, C, sumd] = kmeans(double(Xdash),i,'Distance','sqeuclidean','Replicates',10,'Options',opts,'MaxIter',10);
    cluster_error_pca(count) = sum(sumd);
    if(count == 1)
        figure(3)
        %plot each cluster data
        for j = 1:k
            cluster = reshape(C(j,:),[243,160]);
            subplot(2,4,j),imagesc (cluster); title (['Cluster ',num2str(j)]);
        end
        % plot samples for each cluster
        for j = 1:k
            img_sample = Xdash(idx==j,:);
            img_sample_disp = reshape(img_sample(sample,:),243,160);
            subplot(2,4,4+j),imagesc(img_sample_disp),title(['Sample image for cluster',num2str(j)]);
        end
    end
    count = count+1;
end

figure(2)
subplot(2,1,2),plot(4:15,cluster_error_pca)
xlabel("number of clusters");
ylabel("clustering error");
title("Plot of clustering error over number of clusters after PCA");
