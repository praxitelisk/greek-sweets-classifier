function [net, imdsValidation] = b_cnn_sweets_train(imds)

    %setting the random number to be fixed
    % in order to initialize the cnn's weights
    rng(0);

    
    % [imdsTr, imdsVld] = splitEachLabel(imds, 4000, 'randomize');
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8);

    % cnn design
    batchSize = 10;
    layers=[
        imageInputLayer([224 224 3]);
        convolution2dLayer([3, 3], 16, "Name", "Convolution_1",...
        'Padding','same');
        batchNormalizationLayer
        reluLayer;
        maxPooling2dLayer(2, 'Stride', 2);

        convolution2dLayer([3, 3], 32, "Name", "Convolution_2",...
        'Padding','same');
        batchNormalizationLayer
        reluLayer;
        maxPooling2dLayer([2, 2], 'Stride', 2);

        convolution2dLayer([3, 3], 64, "Name", "Convolution_3",...
        'Padding','same');
        batchNormalizationLayer
        reluLayer;
        maxPooling2dLayer([2, 2], 'Stride', 2);

        convolution2dLayer([3, 3], 128, "Name", "Convolution_4",...
        'Padding','same');
        batchNormalizationLayer
        reluLayer;
        maxPooling2dLayer([2, 2], 'Stride', 2);

        convolution2dLayer([3, 3], 256, "Name", "Convolution_5",...
        'Padding','same');
        batchNormalizationLayer
        reluLayer;
        maxPooling2dLayer(2, 'Stride', 2);

        convolution2dLayer([3, 3], 512, "Name", "Convolution_6",...
        'Padding','same');
        batchNormalizationLayer
        reluLayer;
        maxPooling2dLayer([2, 2], 'Stride', 2);

        fullyConnectedLayer(128, "Name", "fullyConnected_1");
        reluLayer;
        fullyConnectedLayer(4, "Name", "fullyConnected_final"); 
        softmaxLayer;
        classificationLayer;];

    % Cnn training options
    options = trainingOptions('adam', ...
        'InitialLearnRate',0.0001,...
        'LearnRateSchedule', 'piecewise',...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 3, ...
        'ExecutionEnvironment', 'gpu', ...
        'MiniBatchSize', batchSize, ...
        'MaxEpochs',20,...
        'Verbose', false,...
        'Shuffle','every-epoch', ...
        'Plots', 'training-progress',...
        'ValidationData', imdsValidation,...
        'ValidationPatience', 4, ...
        'ValidationFrequency',numel(imdsTrain.Files)/batchSize);

    % training the cnn net
    net = trainNetwork(imdsTrain, layers, options);
end