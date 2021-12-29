clc
clear all
close all

%% Instructions

% This script takes the seperated steady state and penetration data and
% makes a single data file for easy access

% Data.mat contains 2 columns, one of the file name (trial #, type of
% fault) and 1 with all the data points

%% Assemble Call Matrix For Datasets

string_num = string(1:15)';
string_set = ["1.", "2.", "3.", "4."];
classification = [1 2 3 4];
extension = [".txt"];
folder_steady = ["C:\School\MIE697 Int. Man\Class\Final Project\Drill_FaultDataset\Steady State\"];
call_steady = append(folder_steady,string_set,string_num,extension);
steady = reshape(call_steady,[numel(call_steady) 1]);
filename1 = reshape(append(string_set,string_num,extension),[numel(call_steady) 1]);
penetration_set = ["-1"];
folder_penetration = ["C:\School\MIE697 Int. Man\Class\Final Project\Drill_FaultDataset\Penetration\"]
call_penetration = append(folder_penetration,string_set,string_num,penetration_set,extension);
penetration = reshape(call_penetration,[numel(call_penetration) 1]);
filename2 = reshape(append(string_set,string_num,penetration_set,extension),[numel(call_steady) 1]);

%% Assemble Steady State Dataset 

for i = 1:numel(call_steady)

    steadyData{i,1} = filename1(i);
    dat1 = load(steady(i));
    steadyData{i,2} = dat1;
    
end

for i = 1:numel(call_penetration)

    penetrationData{i,1} = filename2(i);
    dat2 = load(penetration(i));
    penetrationData{i,2} = dat2;
    
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
save('Data','steadyData','penetrationData')
