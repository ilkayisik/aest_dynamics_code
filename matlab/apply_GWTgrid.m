%% Calculate the motion energy of videos using GWT 
% (Gabor wavelet transform)
% This piece of code is based on:
% Margalit, E., Biederman, I., Herald, S. B., Yue, X., & von der Malsburg, C. 
% (2016). An applet for the Gabor similarity scaling of the differences 
% between complex stimuli. Attention, Perception, and Psychophysics, 
% 78(8), 2298?2306. https://doi.org/10.3758/s13414-016-1191-7

% uses the function GWTgrid_Simple_Vids
% GWTgrid_SimpleVids(Image,ComplexOrSimple(0=Complex,1=Simple), GridSize,Sigma)

clc
clear all
%% Set paths
% path to the stim diractory where the videos are
stim_dir = '/Users/ilkay.isik/stimuli';
cd(stim_dir)
% add the path of the function [GWTgrid_Simple_Vids]
addpath('/Users/ilkay.isik/aesthetic_dynamics/aest_dynamics_code/matlab')

% list the files in the stim_dir        
files = dir(fullfile(stim_dir, '*.mp4'));

% struct for output data (to put in .csv file)
csvout01 = zeros(numel(files), 1, 900);
csvout02 = zeros(numel(files), 900);
%% Main loop
for i = 1:numel(files)
    
    vid_name = files(i).name
    vr = VideoReader(vid_name);

    % get the dimensions of the video
    size_x = vr.Width;
    size_y = vr.Height;
    % get the number of frames
    nr_frames = vr.NumberOfFrames;

    % frameFreqs = zeros(nr_frames, size_y, size_x);
    V1model = struct('JetsMagnitude', zeros(nr_frames), ...
                     'JetsPhase', zeros(nr_frames), ...
                     'GridPosition', zeros(nr_frames));
    dissim = zeros(1, nr_frames);
    %% Read video and apply the function to every frame
    disp('Applying the GWT function')
    for k = 1:nr_frames
        disp(k)
        frame = read(vr, k);
        % image should be in gray scale 
        frame = rgb2gray(frame);
        [JetsMagnitude, JetsPhase, GridPosition] = GWTgrid_Simple_Vids(frame, 1);
        V1model(k).JetsMagnitude = JetsMagnitude;
        V1model(k).JetsPhase = JetsPhase;
        V1model(k).GridPosition = GridPosition;
    end
    %% Calculate dissimilarity over time
    disp('Calculationg dissimilarity')
    for k = 1:nr_frames - 1
        dissim(k) = norm(reshape(V1model(k + 1).JetsMagnitude, [], 1) - ...
                    reshape(V1model(k).JetsMagnitude, [], 1));
    end

    %% Save to .txt file
    disp('Saving to txt file')
    root_name = strsplit(vid_name, '.');
    ofile_name = strcat(root_name(1), '.txt');
    fileID = fopen(char(ofile_name), 'w');
    fprintf(fileID, '%f\n', dissim(1 : end - 1));
    fprintf(fileID, '%f', dissim(end));
    fclose(fileID);

    %% Save to .csv file
    try
        csvout01(i, :) = dissim(1:900);
    catch ME
        csvout01(i, :) = ones(1, 900) * 9999999;
    end
    
    try
        csvout02(i, :) = dissim(1:900);
    catch ME
        csvout02(i, :) = ones(1, 900) * 9999999;
    end
    
end
% csvwrite('V1_Malsburg_motion_energy.csv', csvout01)
% csvwrite('V1model_motion_energy.csv', csvout02)