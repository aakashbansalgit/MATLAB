[~,~,data] = xlsread('D:\Backup\C\Users\Pelado\Documents\forescast\matlab\TSAF\test folder\Data motores\ET_1.xlsx');
data_mat  = cell2mat(data);
%xlsread('ET_1', 1, 'C:E')
%xlsread(ET_1, 1, 'F:H')
YTrain = (data_mat(:,3))';
XTrain =  (data_mat(:,4:8))';

XTrain = num2cell(XTrain,1);
YTrain = num2cell(YTrain,1);
%%XTrain = num2cell(XTrain,1);
%%YTrain = num2cell(YTrain,1);

miniBatchSize = 1;   %%one predictor sequence

%%Define Network Architecture
numResponses = size(YTrain{1},1);
featureDimension = size(XTrain{1},1);
numHiddenUnits = 500;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(500)  %%50
    dropoutLayer(0.1)  %%0.5
    fullyConnectedLayer(numResponses)
    regressionLayer];


maxepochs = 500;
miniBatchSize = 1;

options = trainingOptions('adam', ...  %%adam
    'MaxEpochs',maxepochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%%Train the Network
net = trainNetwork(XTrain,YTrain,layers,options);

%%Test the Network
[~,~,data] = xlsread('D:\Backup\C\Users\Pelado\Documents\forescast\matlab\TSAF\test folder\Data motores\ET_2_m_input.xlsx');
data_mat  = cell2mat(data);

YTest = (data_mat(:,3))';
XTest =  (data_mat(:,4:8))';

XTest = num2cell(XTest,1);
YTest = num2cell(YTest,1);

net = resetState(net);
YPred = predict(net,XTest)

y1 = (cell2mat(YPred(1:end, 1:end)));  %have to transpose as plot plots columns
plot(y1)
hold on
y2 = (cell2mat(YTest(1:end, 1:end))');
plot(y2)


