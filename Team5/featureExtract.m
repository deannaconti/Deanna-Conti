function features = featureExtract(DataPP,leng,bin_num1,bin_num2)

%% Steady State

bin_num1 = bin_num1;
bin_num2 = bin_num2;

for i = 1:60
    for j = 1:8
    
        new = cell2mat(DataPP.steadyData{i,2}(1,j));
        smnorm=new';
        
        % Time domain

        absmean = sum(abs(smnorm))/length(smnorm);                % Absolute Mean
        maxpeak = max(abs(smnorm));                               % Max Peak
        rms = sqrt((sum(smnorm.^2))/length(smnorm));              % Rms Value
        %srv = ((sum(sqrt(smnorm)))/length(smnorm)).^2;           % Square Root Mean value
        var = sum((smnorm-mean(smnorm)).^2)/(length(smnorm)-1);   % Variance
        kurt = kurtosis(smnorm);                                  % Kurtosis
        crfac = maxpeak/rms;                                      % Crest Factor
        sf = rms/abs(mean(smnorm));                               % Shape Factor
        skew = skewness(smnorm);                                  % Skewness
        FV_time_steady{i,j} = [absmean maxpeak rms var kurt crfac sf skew];
        
    end
end

for i = 1:60
    for j = 1:8
    
        new = cell2mat(DataPP.steadyData{i,2}(1,j));
        smnorm = new';
        % frequency domain features
        for k = 1:8 
        [H1,f] = freqz(smnorm,1,256,25000);
        HH = sum(abs(H1((32*(k-1)+1):(32*k))));
        c(1,k) = HH/sum(abs(H1));
        end
        
        % fft
        
        fs = 32768; % sampling frequency
        n = length(smnorm);
        y = fft(smnorm);
        y = y(1:leng);
        ybins = reshape(y,[(leng/bin_num1),bin_num1]);
        powerbins = sum(abs(ybins).^2/length(ybins));
        power = sum(abs(y).^2/length(y));
        
        %[PSD freq] = FFTfeat(smnorm);
        %PSDbins = reshape(PSD,[(1250/25),25]);
        
        %fft_feat_steady{i,j} = [(powerbins./power)];
        FV_freq_steady{i,j} = [c (powerbins./power)];
    end
end

for i = 1:60
    for j = 1:8
    
        new = cell2mat(DataPP.steadyData{i,2}(1,j));
        smnorm=new';
        clear H1 HH c f;
    
        % Morlet features

        a = 16;
        b = 0.02;
        b1 = 0.5;
        a1 = 0.9;
    
        for t = 1:200
        
            morl(t) = exp(-b1.^2*(t-b).^2/a.^2)*cos(pi*(t-b)/a);
        
        end

        %clear t;clear a*,clear b*;
        morc = conv(smnorm,morl);
        clear morl;
        wvf(1) = std(morc);          % Standard Deviation
        wvf(2) = entropy(morc);      % Wavelet Entropy
        wvf(3) = kurtosis(morc);     % Kurtosis
        wvf(4) = skewness(morc);     % Skewness
        wvf(5) = std(morc).^2;       % Variance = (STD DEV)^2
        FV_wave_steady{i,j} = wvf;
    
    end
end

%% Penetration

for i = 1:60
    for j = 1:8
    
        new = cell2mat(DataPP.penetrationData{i,2}(1,j));
        smnorm=new';
        
        % Time domain

        absmean = sum(abs(smnorm))/length(smnorm);                % Absolute Mean
        maxpeak = max(abs(smnorm));                               % Max Peak
        rms = sqrt((sum(smnorm.^2))/length(smnorm));              % Rms Value
        %srv = ((sum(sqrt(smnorm)))/length(smnorm)).^2;           % Square Root Mean value
        var = sum((smnorm-mean(smnorm)).^2)/(length(smnorm)-1);   % Variance
        kurt = kurtosis(smnorm);                                  % Kurtosis
        crfac = maxpeak/rms;                                      % Crest Factor
        sf = rms/abs(mean(smnorm));                               % Shape Factor
        skew = skewness(smnorm);                                  % Skewness
        FV_time_penetration{i,j} = [absmean maxpeak rms var kurt crfac sf skew];
        
    end
end

for i = 1:60
    for j = 1:8
    
        new = cell2mat(DataPP.penetrationData{i,2}(1,j));
        smnorm = new';
        % frequency domain features
        for k = 1:8 
        [H1,f] = freqz(smnorm,1,256,25000);
        HH = sum(abs(H1((32*(k-1)+1):(32*k))));
        c(1,k) = HH/sum(abs(H1));
        end
        
        % fft
        
        fs = 32768; % sampling frequency
        n = length(smnorm);
        y = fft(smnorm);
        y = y(1:leng);
        ybins = reshape(y,[(leng/bin_num2),bin_num2]);
        powerbins = sum(abs(ybins).^2/length(ybins));
        power = sum(abs(y).^2/length(y));
        
        %[PSD freq] = FFTfeat(smnorm);
        %PSDbins = reshape(PSD,[(1250/25),25]);
        
        %fft_feat_penetration{i,j} = [(powerbins./power)];
        FV_freq_penetration{i,j} = [c (powerbins./power)];
    end
end

for i = 1:60
    for j = 1:8
    
        new = cell2mat(DataPP.steadyData{i,2}(1,j));
        smnorm=new';
        clear H1 HH c f;
    
        % Morlet features

        a = 16;
        b = 0.02;
        b1 = 0.5;
        a1 = 0.9;
    
        for t = 1:200
        
            morl(t) = exp(-b1.^2*(t-b).^2/a.^2)*cos(pi*(t-b)/a);
            
        end

        clear t;clear a*,clear b*;
        morc = conv(smnorm,morl);
        clear morl;
        wvf(1) = std(morc);          % Standard Deviation
        wvf(2) = entropy(morc);      % Wavelet Entropy
        wvf(3) = kurtosis(morc);     % Kurtosis
        wvf(4) = skewness(morc);     % Skewness
        wvf(5) = std(morc).^2;       % Variance = (STD DEV)^2
        FV_wave_penetration{i,j} = wvf;
    
    end
end

%for i = 1:60
    %for j = 1:8
    
        %new = cell2mat(data.steadyData{i,2}(1,j));
        %smnorm=new';
     
        % DCT 

        %X = dct(smnorm);
        %[XX,ind] = sort(abs(X),'descend');
        %i = 1;
        %while norm(X(ind(1:i)))/norm(X) < 0.995
            %i = i + 1;
        %end
        %needed = i;
        
           
        %DCT_wave_penetration{i,j} = wvf;
    
    %end
%end

save('features', 'FV_freq_penetration', 'FV_freq_steady', 'FV_time_penetration', 'FV_time_steady', 'FV_wave_penetration', 'FV_wave_steady')

end