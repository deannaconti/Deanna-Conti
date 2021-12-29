function [scoreCal, percentage, sum_percentage] = PCAfinal(features)

%% Normalize

[ZscoreCal,muCal,stdCal] = zscore(features);         % normalize and centralize data, record mean and SD             

%% PCA

[coeffCal,scoreCal,latentCal] = pca(ZscoreCal);    % PCA

percentage = latentCal*100/sum(latentCal);         % percentage of principle components on variability
sum_percentage = cumsum(percentage);               % find cumulative percentage

%pareto(percentage);
bar(percentage);
xlabel('Principal Component')
ylabel('Variance Explained (%)')

end
