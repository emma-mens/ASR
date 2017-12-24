function [] = get_spectrogram( path, outfile )
    
    WINDOW_SIZE = 2048;
    OVERLAP = 512;
    
    % Load Mixed song (two channels: left and right)
    
    [mix, ~] = audioread(path);
    
    s = size(mix);
    if s(2) == 2
        % Two channels
        mixl = mix(:,1);
        % mixr = mix(:,2);
        
        % Compute Short Time Fourier Transform (STFT), with hanning window
        mixl_spect = spectrogram(mixl, hann(WINDOW_SIZE), OVERLAP);
        % mixr_spect = spectrogram(mixr, hann(WINDOW_SIZE), OVERLAP);
        
        mixl_mag = abs(mixl_spect); mixl_phase = angle(mixl_spect);
        % mixr_mag = abs(mixr_spect); mixr_phase = angle(mixr_spect);
        
        % Save phase and magnitude matrices to file

        % Save left channel
        
        lphase_file = strcat(outfile, '_phase.txt');
        lmag_file = strcat(outfile, '_mag.txt');
        csvwrite(lphase_file, mixl_phase);
        csvwrite(lmag_file, mixl_mag);
        
        % Save right channel
        
        % rphase_file = strcat(outfile, '_right_phase.txt');
        % rmag_file = strcat(outfile, '_right_mag.txt');
        % csvwrite(rphase_file, mixr_phase);
        % csvwrite(rmag_file, mixr_mag);
        
    else
        % One channel
        mix_spect = spectrogram(mix, hann(WINDOW_SIZE), OVERLAP);

        mix_mag = abs(mix_spect); mix_phase = angle(mix_spect);

        % Save phase and magnitude matrices to file
%         file = strsplit(path, '.');
        phase_file = strcat(outfile, 'phase.txt');
        mag_file = strcat(outfile, 'mag.txt');

        csvwrite(phase_file, mix_phase);
        csvwrite(mag_file, mix_mag); 
    end
    quit
end

