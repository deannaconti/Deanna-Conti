function num = dataPreProcess(Data,n)

%% Instructions

% This script takes the raw data and preprocesses it based on the steps in
% the papee

% DataPP.mat contains 2 columns, one of the file name (trial #, type of
% fault) and 1 with all the data points

%% Load Data
%Data = load('Data.mat');

%% Pre-Processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Steady State Data

% Low Pass Butterworth Filter

for i = 1:length(Data.steadyData);

data1 = Data.steadyData{i,2};

fc = 12000;
fs = 32768;
[a,b] = butter(20,fc/(fs/2));
outsignal = filter(a,b,data1);

%Data.steadyData{i,2} = outsignal;

% Clipping, Moving Avg, Normalization

%time_int = 8;
%pts = length(data1)/time_int;
%time_int = buffer(outsignal,pts);
time_int = reshape(outsignal,length(outsignal)/8,[]);

for j = 1:8

% Clipping    
    
olap{j} = buffer(time_int(:,j),8192,4681,'nodelay');
stand{j} = std(olap{j});
[m(j) I(j)] = min(stand{1,j});
clipped{j} = olap{j}(:,I(j));

% figure(88)
% plot(clipped{j})

% 10 pt Moving Average

coeff = 1/n*ones(n,1);

averaged{j} = filter(coeff,1,clipped{j});

% figure(77)
% plot(clipped{j})
% hold on
% plot(averaged{j})
% legend('Clipped Data','Moving Avg. Data')


% Max Min Normalization

normal{j} = normalize(averaged{j},1,'range');

% figure(66)
% plot(normal{j})

end

Data.steadyData{i,2} = normal;

end

%% Penetration Data

clear a averaged b clipped coeff data1 fc fs i j I m normal olap outsignal pts stand time_int

for i = 1:length(Data.penetrationData);

data1 = Data.penetrationData{i,2};

fc = 12000;
fs = 32768;
[a,b] = butter(20,fc/(fs/2));
outsignal = filter(a,b,data1);

%Data.penetrationData{i,2} = outsignal;

% Clipping, Moving Avg, Normalization

%time_int = 8;
%pts = length(data1)/time_int;
%time_int = buffer(outsignal,pts);
time_int = reshape(outsignal,length(outsignal)/8,[]);

for j = 1:8

% Clipping    
    
olap{j} = buffer(time_int(:,j),8192,4681,'nodelay');
stand{j} = std(olap{j});
[m(j) I(j)] = min(stand{1,j});
clipped{j} = olap{j}(:,I(j));

% n pt Moving Average
coeff = 1/n*ones(n,1);
averaged{j} = filter(coeff,1,clipped{j});

% Max Min Normalization

normal{j} = normalize(averaged{j},'range');

end

Data.penetrationData{i,2} = normal;

end

for i = 1:15

    steadyData{i,3} = 1;
    steadyData{i+15,3} = 2;
    steadyData{i+30,3} = 3;
    steadyData{i+45,3} = 4;
    
    penetrationData{i,3} = 1;
    penetrationData{i+15,3} = 2;
    penetrationData{i+30,3} = 3;
    penetrationData{i+45,3} = 4;

end

%% Save Data
penetrationData = Data.penetrationData;
steadyData = Data.steadyData;
save('DataPP','steadyData','penetrationData')

end

