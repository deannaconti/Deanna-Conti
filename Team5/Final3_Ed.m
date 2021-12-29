clc 
clear
close all
tic
%% User Inputs 

n = 2;       % # of averaging windows
leng = 1500; % cut off frequncy
bin_num1 = 100; % # of bins
bin_num2 = 100; % ust be same as bin_num1


Data = load('Data.mat');
dataPreProcess(Data,n);                             % output is DataPP.mat
DataPP = load('DataPP.mat');
featureExtract(DataPP,leng,bin_num1,bin_num2);      % output is feautures

data_features = load('features.mat');

%% User Inputs

method = 4;
k = 8; % # of k folds
iter = 5; %iterations for use in Ensemble

%% Assemble Features

% Frequency
features_penetration_f = reshape(data_features.FV_freq_penetration',1,numel(data_features.FV_freq_penetration))';
features_steady_f = reshape(data_features.FV_freq_steady',1,numel(data_features.FV_freq_steady))';
features_f = cell2mat([features_steady_f; features_penetration_f]);

% Time and frequency 
%features_penetration = reshape([cell2mat(data_features.FV_freq_penetration) cell2mat(data_features.FV_time_penetration)],[480,66]);
%features_steady = reshape([cell2mat(data_features.FV_freq_steady) cell2mat(data_features.FV_time_steady)],[480,66]);
%features = ([features_steady; features_penetration]);

features = [features_f]; % features_t];


%% PCA

[scoreCal, percentage, sum_percentage] = PCAfinal(features);
PCdata = scoreCal; 

% figure
% scatter3(scoreCal(1:120,1),scoreCal(1:120,2),scoreCal(1:120,3),'r.');
% hold on
% scatter3(scoreCal(121:240,1),scoreCal(121:240,2),scoreCal(121:240,3),'b.');
% scatter3(scoreCal(241:360,1),scoreCal(241:360,2),scoreCal(241:360,3),'g.');
% scatter3(scoreCal(361:480,1),scoreCal(361:480,2),scoreCal(361:480,3),'m^');
% scatter3(scoreCal(481:600,1),scoreCal(481:600,2),scoreCal(481:600,3),'r^');
% scatter3(scoreCal(601:720,1),scoreCal(601:720,2),scoreCal(601:720,3),'b^');
% scatter3(scoreCal(721:840,1),scoreCal(721:840,2),scoreCal(721:840,3),'g^');
% scatter3(scoreCal(841:960,1),scoreCal(841:960,2),scoreCal(841:960,3),'m^');
% xlabel('PC 1')
% ylabel('PC 2')
% zlabel('PC 3')
% legend('Steady State D1', 'Steady State D2', 'Steady State D3', 'Steady State D4', 'Penetration D1', 'Penetration D2', 'Penetration D3', 'Penetration D4')

%% Assemble Data and Labels

GT1SS = ones(120);
GT1SS = GT1SS(:,1);
PCdata1SS = [GT1SS PCdata(1:120,:)];

GT2SS = 2*ones(120);
GT2SS = GT2SS(:,1);
PCdata2SS = [GT2SS PCdata(121:240,:)];

GT3SS = 3*ones(120);
GT3SS = GT3SS(:,1);
PCdata3SS = [GT3SS PCdata(241:360,:)];

GT4SS = 4*ones(120);
GT4SS = GT4SS(:,1);
PCdata4SS = [GT4SS PCdata(361:480,:)];

GT1P = ones(120);
GT1P = GT1P(:,1);
PCdata1P = [GT1P PCdata(481:600,:)];

GT2P = 2*ones(120);
GT2P = GT2P(:,1);
PCdata2P = [GT2P PCdata(601:720,:)];

GT3P = 3*ones(120);
GT3P = GT3P(:,1);
PCdata3P = [GT3P PCdata(721:840,:)];

GT4P = 4*ones(120);
GT4P = GT4P(:,1);
PCdata4P = [GT4P PCdata(841:960,:)];

%% Define k-Fold

hpartition = cvpartition(length(PCdata(1:120,1)),'kFold',k); 

%% Classification 

label_SVM = zeros(120,iter);
label_ANN = zeros(120,iter);
pc_count = 1; % initialize counting
axis = [5 10 20 35 55 80 108];
for ACC_count = 1:3
for PC = axis
    tic
for iteration = 1:iter
    
    %%%%%% ANN %%%%%%
    
for i = 1:k
    
    % Steady State
idxTrain1SS = training(hpartition,i);            % get indexes of training data for Drill 1 SS
XTRAIN1SS = PCdata1SS(idxTrain1SS,:);              % return training data for Drill 1 SS
idxTest1SS = test(hpartition,i);                 % get indexes of testing data for Drill 1 SS
XTEST1SS = PCdata1SS(idxTest1SS,:);                % return testing data for Drill 1 SS

idxTrain2SS = training(hpartition,i);            % get indexes of training data for Drill 2 SS
XTRAIN2SS = PCdata2SS(idxTrain2SS,:);              % return training data for Drill 2 SS
idxTest2SS = test(hpartition,i);                 % get indexes of testing data for Drill 2 SS
XTEST2SS = PCdata2SS(idxTest2SS,:);                % return testing data for Drill 2 SS

idxTrain3SS = training(hpartition,i);            % get indexes of training data for Drill 3 SS
XTRAIN3SS = PCdata3SS(idxTrain3SS,:);              % return training data for Drill 3 SS
idxTest3SS = test(hpartition,i);                 % get indexes of testing data for Drill 3 SS
XTEST3SS = PCdata3SS(idxTest3SS,:);                % return testing data for Drill 3 SS

idxTrain4SS = training(hpartition,i);            % get indexes of training data for Drill 4 SS
XTRAIN4SS = PCdata4SS(idxTrain4SS,:);              % return training data for Drill 4 SS
idxTest4SS = test(hpartition,i);                 % get indexes of testing data for Drill 4 SS
XTEST4SS = PCdata4SS(idxTest4SS,:);                % return testing data for Drill 4 SS

% Penetration

idxTrain1P = training(hpartition,i);            % get indexes of training data for Drill 1 SS
XTRAIN1P = PCdata1P(idxTrain1P,:);              % return training data for Drill 1 SS
idxTest1P = test(hpartition,i);                 % get indexes of testing data for Drill 1 SS
XTEST1P = PCdata1P(idxTest1P,:);                % return testing data for Drill 1 SS

idxTrain2P = training(hpartition,i);            % get indexes of training data for Drill 2 SS
XTRAIN2P = PCdata2SS(idxTrain2P,:);              % return training data for Drill 2 SS
idxTest2P = test(hpartition,i);                 % get indexes of testing data for Drill 2 SS
XTEST2P = PCdata2SS(idxTest2P,:);                % return testing data for Drill 2 SS

idxTrain3P = training(hpartition,i);            % get indexes of training data for Drill 3 SS
XTRAIN3P = PCdata3P(idxTrain3P,:);              % return training data for Drill 3 SS
idxTest3P = test(hpartition,i);                 % get indexes of testing data for Drill 3 SS
XTEST3P = PCdata3P(idxTest3P,:);                % return testing data for Drill 3 SS

idxTrain4P = training(hpartition,i);            % get indexes of training data for Drill 4 SS
XTRAIN4P = PCdata4P(idxTrain4P,:);              % return training data for Drill 4 SS
idxTest4P = test(hpartition,i);                 % get indexes of testing data for Drill 4 SS
XTEST4P = PCdata4P(idxTest4P,:);                % return testing data for Drill 4 SS

% PC = 105;

TRAIN = [XTRAIN1SS(:,2:PC+1); XTRAIN2SS(:,2:PC+1); XTRAIN3SS(:,2:PC+1); XTRAIN4SS(:,2:PC+1); XTRAIN1P(:,2:PC+1); XTRAIN2P(:,2:PC+1); XTRAIN3P(:,2:PC+1); XTRAIN4P(:,2:PC+1)];
TEST = [XTEST1SS(:,2:PC+1); XTEST2SS(:,2:PC+1); XTEST3SS(:,2:PC+1); XTEST4SS(:,2:PC+1); XTEST1P(:,2:PC+1); XTEST2P(:,2:PC+1); XTEST3P(:,2:PC+1); XTEST4P(:,2:PC+1)];
GTTRAIN = [XTRAIN1SS(:,1); XTRAIN2SS(:,1); XTRAIN3SS(:,1); XTRAIN4SS(:,1); XTRAIN1P(:,1); XTRAIN2P(:,1); XTRAIN3P(:,1); XTRAIN4P(:,1)];
GTTEST = [XTEST1SS(:,1); XTEST2SS(:,1); XTEST3SS(:,1); XTEST4SS(:,1); XTEST1P(:,1); XTEST2P(:,1); XTEST3P(:,1); XTEST4P(:,1)];
    
% Train ANN Model
GTTRAIN = (GTTRAIN);
ANNModel = fitcnet(TRAIN,GTTRAIN);

% Predict ANN Test Data
label_ANN(:,iteration) = predict(ANNModel,TEST);

[m,order] = confusionmat(GTTEST,label_ANN(:,iteration));
tbl{i} = m;
ord{i} = order;
acc_ANN(i) = ((m(1,1)+m(2,2)+m(3,3)+m(4,4))/((m(1,2)+m(1,3)+m(1,4)+m(2,1)+m(2,3)+m(2,4)+m(3,1)+m(3,2)+m(3,4)+m(4,1)+m(4,2)+m(4,3))+(m(1,1)+m(2,2)+m(3,3)+m(4,4))));
    
end
    
Accuracy_ANN = mean(acc_ANN);
    
ACCURACY(pc_count,1,ACC_count) = Accuracy_ANN;

% figure(98)
% for i = 1:k
% subplot(2,5,i)
% cm = confusionchart(tbl{i},ord{i});
% sgtitle('ANN 1->10')
% end
end
    
for i = 1:k %%%%% SVM %%%%%

% Train SVM Model
GTTRAIN = (GTTRAIN);
SVMModel = fitcecoc(TRAIN,GTTRAIN);

% Predict SVM Test Data
label_SVM(:,iteration) = predict(SVMModel,TEST);

% Generate Confusion Matrix
[m,order] = confusionmat(GTTEST,label_SVM(:,iteration));
tbl{i} = m;
ord{i} = order;

    %TN = m(2,2);
    %TP = m(1,1);
    %FN = m(1,2);
    %FP = m(2,1);
    
    %Accuracy(i) = (TN+TP)/length(TEST);
    
    CVMdl = crossval(SVMModel);
    genError = kfoldLoss(CVMdl);
    acc(i) = 1-genError;
    acc_SVM(i) = ((m(1,1)+m(2,2)+m(3,3)+m(4,4))/((m(1,2)+m(1,3)+m(1,4)+m(2,1)+m(2,3)+m(2,4)+m(3,1)+m(3,2)+m(3,4)+m(4,1)+m(4,2)+m(4,3))+(m(1,1)+m(2,2)+m(3,3)+m(4,4))));
end
    
    %Performance = [Accuracy]
    average = mean(acc);
    Accuracy_SVM = mean(acc_SVM);
    
    ACCURACY(pc_count,2,ACC_count) = Accuracy_SVM;
    
% figure(96)
% for i = 1:k
% subplot(2,5,i)
% cm = confusionchart(tbl{i},ord{i});
% sgtitle('SVM Confusion Matrices 1->10')
% end



%%%%% KNN %%%%%

 for i = 1:k

     hpartition = cvpartition(length(PCdata(1:120,1)),'kFold',k); 

   
    md = fitcknn(TRAIN,GTTRAIN,'NumNeighbors',4,'Standardize',1);
%     figure
%     gscatter(TRAIN(:,1),TRAIN(:,2),GTTRAIN,'rgbm','osd');
%     xlabel('Dimension 1')
%     ylabel('Dimension 2')
%     hold off
%     
    % Predict KNN Test Data and Predict Confusion Matrix
  
    [label_KNN,score,cost] = predict(md,TEST);
    %label = str2double(label);
    [m,order] = confusionmat(GTTEST,label_KNN);
    tbl{i} = m;
    ord{i} = order;
    acc_KNN(i) = ((m(1,1)+m(2,2)+m(3,3)+m(4,4))/((m(1,2)+m(1,3)+m(1,4)+m(2,1)+m(2,3)+m(2,4)+m(3,1)+m(3,2)+m(3,4)+m(4,1)+m(4,2)+m(4,3))+(m(1,1)+m(2,2)+m(3,3)+m(4,4))));
 end
    
    Accuracy_KNN = mean(acc_KNN);
     
    ACCURACY(pc_count,3,ACC_count) = Accuracy_KNN;
    
% figure(97)
% for i = 1:k
% subplot(2,5,i)
% cm = confusionchart(tbl{i},ord{i});
% sgtitle('KNN Confusion Matrices 1->10')
% end
    

%%%%%% Random Forest %%%%%%
    
for i = 1:k

% Train Random Forest Model
GTTRAIN = (GTTRAIN);
TREE = fitctree(TRAIN,GTTRAIN,'MaxNumSplits',5000);

% Predict Random Forest Test Data
label_RF = predict(TREE,TEST);


[m,order] = confusionmat(GTTEST,label_RF);
tbl{i} = m;
ord{i} = order;
acc_RF(i) = ((m(1,1)+m(2,2)+m(3,3)+m(4,4))/((m(1,2)+m(1,3)+m(1,4)+m(2,1)+m(2,3)+m(2,4)+m(3,1)+m(3,2)+m(3,4)+m(4,1)+m(4,2)+m(4,3))+(m(1,1)+m(2,2)+m(3,3)+m(4,4))));
end
    
Accuracy_RF = mean(acc_RF);
    
ACCURACY(pc_count,4,ACC_count) = Accuracy_RF;

% figure(99)
% for i = 1:k
% subplot(2,5,i)
% cm = confusionchart(tbl{i},ord{i});
% sgtitle('Random Forest 1->10')
% end

%%%%% Ensemble %%%%%%

% for j = 1:iter
%     LABELS(:,j) = label_ANN(:,j);
% end

LABELS_all = [label_ANN(:,end), label_KNN, label_RF, label_SVM(:,end)];
MODE_all = mode(LABELS_all');
label_Ensemble_all = MODE_all';
[m,~] = confusionmat(GTTEST,label_Ensemble_all);
acc_Ensemble_all = ((m(1,1)+m(2,2)+m(3,3)+m(4,4))/((m(1,2)+m(1,3)+m(1,4)+m(2,1)+m(2,3)+m(2,4)+m(3,1)+m(3,2)+m(3,4)+m(4,1)+m(4,2)+m(4,3))+(m(1,1)+m(2,2)+m(3,3)+m(4,4))));
Accuracy_Ensemble_all = mean(acc_Ensemble_all);
ACCURACY(pc_count,5,ACC_count) = Accuracy_Ensemble_all;

% for q = 1:iter
% LABELS_ANN(:,q) = label_ANN(:,q);
% end
MODE_ANN = mode(label_ANN');
label_Ensemble_ANN = MODE_ANN';
[m,~] = confusionmat(GTTEST,label_Ensemble_ANN);
acc_Ensemble_ANN = ((m(1,1)+m(2,2)+m(3,3)+m(4,4))/((m(1,2)+m(1,3)+m(1,4)+m(2,1)+m(2,3)+m(2,4)+m(3,1)+m(3,2)+m(3,4)+m(4,1)+m(4,2)+m(4,3))+(m(1,1)+m(2,2)+m(3,3)+m(4,4))));
Accuracy_Ensemble_ANN = mean(acc_Ensemble_ANN);
ACCURACY(pc_count,6,ACC_count) = Accuracy_Ensemble_ANN;

pc_count = pc_count + 1;
toc
end


figure
plot(axis,ACCURACY(:,:,ACC_count),'-*')
xlabel('Features Retained')
ylabel('Accuracy')
legend('ANN','SVM','KNN','Random Forest','Ensemble All','Ensemble ANN','location','southeast')

pc_count = 1;
end
toc

CLASS_DATA = mean(ACCURACY,3); %Classification accuracy averaged over 3 trials
figure
plot(axis,CLASS_DATA,'-*','linewidth',3,'markersize',4)
xlabel('Features Retained')
ylabel('Accuracy')
legend('ANN','SVM','KNN','Random Forest','Ensemble All','Ensemble ANN','location','southeast')

