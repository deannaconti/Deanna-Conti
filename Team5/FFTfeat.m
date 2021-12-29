function [PSD freq] = FFTfeat(data)

%% Plot Raw Data
f = load('DataPP.mat');
f = cell2mat(f.steadyData{1,2}(1));
%f = f';
t = linspace(0,1/(32768/8192),length(f));                  % create time vector of 8s
dt = 0.25/(length(f))                             % 1/(Sampling Frequency)

%% Compute the Fast Fourier Transform FFT
n = length(t);
y = fft(f); % Compute the fast Fourier transform
PSD = y.*conj(y)/n; % Power spectrum (power per freq)
freq = 1/(dt*n)*(0:n); % Create x-axis of frequencies in Hz
L = 1:1250;

PSD = PSD(L);
freq = freq(L);


end
