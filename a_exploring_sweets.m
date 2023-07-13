function [imds, averageHistograms] = a_exploring_sweets()
    
    %setting the random number to be fixed
    % in order to initialize the cnn's weights
    rng(0);

    %setting up the pathname
    data = "C:\PRAXIDATA\Praxitelis\2.2 - Master\Semester 2\CNN_Sweets";

    % reading the images
    imds = imageDatastore(data, 'IncludeSubFolders',true,...
        "FileExtensions",[".jpg",".tif"],...
        'LabelSource','foldernames');

    % useful info regarding some images
    % some statistics.
    numObs = length(imds.Labels);
    numObsPerClass = countEachLabel(imds)
    figure;
    histogram(imds.Labels)
    set(gca,'TickLabelInterpreter','none')


    % calculate the histogram of average RGB colors for all images
    numBins = 256; % Number of bins in the histogram
    accumulatedHistograms = zeros(numBins, 3); % One column for each RGB channel
    imageCount = 0;
    
    
    while hasdata(imds)
        img = read(imds);
        % Split the image into RGB channels
        redChannel = img(:,:,1);
        greenChannel = img(:,:,2);
        blueChannel = img(:,:,3);

        % Calculate histograms for each channel
        redHistogram = imhist(redChannel, numBins);
        greenHistogram = imhist(greenChannel, numBins);
        blueHistogram = imhist(blueChannel, numBins);

        % Accumulate histograms
        accumulatedHistograms(:, 1) = accumulatedHistograms(:, 1) + redHistogram;
        accumulatedHistograms(:, 2) = accumulatedHistograms(:, 2) + greenHistogram;
        accumulatedHistograms(:, 3) = accumulatedHistograms(:, 3) + blueHistogram;

        imageCount = imageCount + 1;
    end
    
    averageHistograms = accumulatedHistograms / imageCount;

    
    figure;
    subplot(1, 3, 1);
    bar(0:numBins-1, averageHistograms(:, 1), 'r');
    title('Red Channel');
    xlabel('Pixel Value');
    ylabel('Frequency');

    subplot(1, 3, 2);
    bar(0:numBins-1, averageHistograms(:, 2), 'g');
    title('Green Channel');
    xlabel('Pixel Value');
    ylabel('Frequency');

    subplot(1, 3, 3);
    bar(0:numBins-1, averageHistograms(:, 3), 'b');
    title('Blue Channel');
    xlabel('Pixel Value');
    ylabel('Frequency');

    sgtitle('Average Histograms of RGB Channels');
end