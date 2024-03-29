function [JetsMagnitude, JetsPhase, GridPosition] = GWTgrid_Simple_Vids(Im, ComplexOrSimple, GridSize, Sigma)

%% GWT: Gabor wavelet transform
%% The goal of this function is to transform an image with a Gabor wavelet
%% method, and then convolution values at limited positions of the image
%% will be chosen as output
%%
%% Usage: [JetsMagnitude, JetsPhase, GridPosition] = GWTgrid_Simple_Vids(Im, ComplexOrSimple, GridSize, Sigma)
%%
%% Inputs to the function:
%%   Im                  -- The image you want to reconstruct with this function
%%
%%   ComplexOrSimple     -- If input is 0, the JetsMagnitude would be complex cell responses (40 values) (default)
%%                          If input is 1, the JetsMagnitude would be simple cell responses (80 values)
%%
%%   GridSize            -- If input is 0, grid size is 10*10 (default);
%%                          If input is 1, grid size is 12*12 ;
%%                          If input is 2, grid size would be the image size (128*128 or 256*256)
%%
%%   Sigma               -- Control the size of Gaussian envelope
%%
%%
%% Outputs of the functions:
%%   JetsMagnitude       -- Gabor wavelet transform magnitude
%%   JetsPhase           -- Gabor wavelet transform phase
%%   GridPosition        -- positions sampled
%%
%% Created by Xiaomin Yue at 7/25/2004
%%
%% Last updated: 24.01.2018
%%
dbstop if error

if nargin < 1
    disp('Please provide an input image.');
    return;
end

if nargin < 2
    ComplexOrSimple = 0;
    GridSize = 0;
    Sigma = 2*pi;
end

if nargin < 3
    GridSize = 0;
    Sigma = 2*pi;
end

if nargin < 4
    Sigma = 2*pi;
end

%% FFT of the image
Im = double(Im);
ImFreq = fft2(Im);
[SizeY, SizeX] = size(Im);

%% generate the grid
if SizeX == 256
    if GridSize == 0
        RangeXY = 40:20:220;
    elseif GridSize == 1
        RangeXY = 20:20:240;
    else
        RangeXY = 1:256;
    end    
    [xx, yy] = meshgrid(RangeXY, RangeXY);
    Grid = xx + yy * i;
    Grid = Grid(:);
    
elseif SizeX == 128
    if GridSize == 0
        RangeXY = 20:10:110;
    elseif GridSize == 1;
        RangeXY = 10:10:120;
    else
        RangeXY = 1:128;
    end    
    [xx, yy] = meshgrid(RangeXY, RangeXY);
    Grid = xx + yy * i;
    Grid = Grid(:);
    
elseif SizeX == 1280 && SizeY == 720 % for our video frames
    RangeX = 10:10:1270;
    RangeY = 10:10:710; % what should be the grid size?
    [xx, yy] = meshgrid(RangeX, RangeY);
    Grid = xx + yy * i; % creates one (complex, 2D) matrix
    Grid = Grid(:); % transforms matrix into one single column vector
else
    disp('The image has to be 256x256 or 128x128. Please try again.');
    return;
end
GridPosition = [imag(Grid) real(Grid)];

%% setup the parameters
nScale = 5; nOrientation = 8;
xyResL = SizeX; xHalfResL = SizeX/2; yHalfResL = SizeY/2;
kxFactor = 2*pi/xyResL;
kyFactor = 2*pi/xyResL;

%% setup space coordinates
[tx, ty] = meshgrid(-xHalfResL:xHalfResL-1, -yHalfResL:yHalfResL-1);        
tx = kxFactor * tx;
ty = kyFactor * (-ty);

%% initialise useful variables
if ComplexOrSimple == 0
    JetsMagnitude  = zeros(length(Grid), nScale * nOrientation);
    JetsPhase      = zeros(length(Grid), nScale * nOrientation);
else
    JetsMagnitude  = zeros(length(Grid), 2 * nScale * nOrientation);
    JetsPhase      = zeros(length(Grid), nScale * nOrientation);
end

for LevelL = 0:nScale - 1
    k0 = (pi / 2) * (1 / sqrt(2)) ^ LevelL;
    for DirecL = 0:nOrientation - 1
        kA = pi * DirecL / nOrientation;
        k0X = k0 * cos(kA);
        k0Y = k0 * sin(kA);
        %% generate a kernel specified scale and orientation, which has 
        %  DC on the center
        FreqKernel = 2*pi*(exp(-(Sigma/k0)^2/2*((k0X-tx).^2+(k0Y-ty).^2))...
                     -exp(-(Sigma/k0)^2/2*(k0^2+tx.^2+ty.^2)));
        %% use fftshift to change DC to the corners
        FreqKernel = fftshift(FreqKernel);
        
        %% Convolve the image with a kernel specified scale and orientation
        TmpFilterImage = ImFreq.*FreqKernel;
        %% calculate magnitude and phase
        if ComplexOrSimple == 0
            TmpGWTMag = abs(ifft2(TmpFilterImage));
            TmpGWTPhase = angle(ifft2(TmpFilterImage));
            %% get magnitude and phase at specific positions
            tmpMag = TmpGWTMag(RangeX, RangeY);
            tmpMag = (tmpMag');
            JetsMagnitude(:, LevelL * nOrientation + DirecL+1) = tmpMag(:);
            tmpPhase = TmpGWTPhase(RangeX, RangeY);
            tmpPhase = (tmpPhase') + pi;
            JetsPhase(:, LevelL * nOrientation + DirecL+1) = tmpPhase(:);
        else
            TmpGWTMag_real = (real(ifft2(TmpFilterImage)));
            TmpGWTMag_imag = (imag(ifft2(TmpFilterImage)));
            TmpGWTPhase = angle(ifft2(TmpFilterImage));
            %% get magnitude and phase at specific positions
            tmpMag_real = TmpGWTMag_real(RangeY,RangeX);
            tmpMag_real = (tmpMag_real');
            tmpMag_imag = TmpGWTMag_imag(RangeY,RangeX);
            tmpMag_imag = (tmpMag_imag');
            JetsMagnitude_real(:, LevelL * nOrientation + DirecL + 1) = ...
                               tmpMag_real(:);
            JetsMagnitude_imag(:, LevelL * nOrientation + DirecL + 1) = ...
                               tmpMag_imag(:);
            tmpPhase = TmpGWTPhase(RangeY, RangeX);
            tmpPhase = (tmpPhase') + pi;
            JetsPhase(:, LevelL * nOrientation + DirecL + 1) = tmpPhase(:);            
        end    
    end
end    

if ComplexOrSimple ~= 0
    JetsMagnitude = [JetsMagnitude_real JetsMagnitude_imag];
end
    
end % function end