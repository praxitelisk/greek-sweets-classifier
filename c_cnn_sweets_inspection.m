function c_cnn_sweets_inspection(net, imdsValidation)

    % inspecting the cnn net upon the validation set
    Prdigits = classify(net, imdsValidation);
    digits = imdsValidation.Labels;
    accuracy = 100 * sum(Prdigits==digits)/numel(digits);
    disp("accuracy after training on validation set: " + accuracy +"%")

    %plotting the confussion matrix
    plotconfusion(imdsValidation.Labels, Prdigits)

    % analyzing the network
    analyzeNetwork(net)


    %Visualize Features of a Convolutional Neural Network
    % Convolution layer 1
    layer = 2;
    name = net.Layers(layer).Name;
    channels = 1:16;
    I = deepDreamImage(net,name,channels, ...
        'PyramidLevels',1);

    figure
    I = imtile(I,'ThumbnailSize',[64 64]);
    imshow(I)
    title(['Layer ',name,' Features'],'Interpreter','none')

    % Convolution layer 2
    layer = 6;
    name = net.Layers(layer).Name;
    channels = 1:32;
    I = deepDreamImage(net,name,channels, ...
        'PyramidLevels',1);

    figure
    I = imtile(I,'ThumbnailSize',[64 64]);
    imshow(I)
    title(['Layer ',name,' Features'],'Interpreter','none')

    % Convolution layer 3
    layer = 10;
    name = net.Layers(layer).Name;
    channels = 1:64;
    I = deepDreamImage(net,name,channels, ...
        'PyramidLevels',1);

    figure
    I = imtile(I,'ThumbnailSize',[64 64]);
    imshow(I)
    title(['Layer ',name,' Features'],'Interpreter','none')

    % Convolution layer 4
    layer = 14;
    name = net.Layers(layer).Name;
    channels = 1:128;
    I = deepDreamImage(net,name,channels, ...
        'PyramidLevels',1);

    figure
    I = imtile(I,'ThumbnailSize',[64 64]);
    imshow(I)
    title(['Layer ',name,' Features'],'Interpreter','none')

    % Convolution layer 5
    layer = 18;
    name = net.Layers(layer).Name;
    channels = 1:256;
    I = deepDreamImage(net,name,channels, ...
        'PyramidLevels',1);

    figure
    I = imtile(I,'ThumbnailSize',[64 64]);
    imshow(I)
    title(['Layer ',name,' Features'],'Interpreter','none')
    
    % Convolution layer 6
    % cannot visualize features due to current GPU's
    % memory limitations (upto 2GB)
    
    % last full connected layer
    layer = 28;
    name = net.Layers(layer).Name;
    channels = 1:4;
    I = deepDreamImage(net,name,channels, ...
    'Verbose',true, ...
    'NumIterations',100, ...
    'PyramidLevels',2);
    
    figure
    I = imtile(I,'ThumbnailSize',[64 64]);
    imshow(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])


end