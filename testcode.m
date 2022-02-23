data = readtable('inputdata.xlsx')
data = table2array(data)
rec = length(data(:,1))
VarNum = length(data(1,:))
Y = data(:,VarNum)
data( : , end) = [ ]
numTimeStepsTrain = floor(0.9*rec);
dataTrain = data(1:numTimeStepsTrain+1,:);
dataTest = data(numTimeStepsTrain+1:end,:);
dataTrain
dataTest
%   Normalize sales_price to a value between 0 and 1 (Training Data Set)
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
XTrain
YTrain
