clear all;
addpath('release_NBSGD');

% Hingle and Logistic losses are for classification
% L2, L1 and Eps-Intensitive losses are for regression


% binary classification
fprintf('BinaryClass ');

[yyTrain, xxTrain]=libsvmread('data\mushrooms');
%[yyTrain, xxTrain]=libsvmread('data\a9a');[yyTest, xxTest]=libsvmread('data\a9a.t');

if isempty(xxTrain)
    return;
end

if ~exist('xxTest','var')
    % create test set if it is not existed
    idx=randperm(length(yyTrain));
    xxTest=xxTrain(idx(1:ceil(0.1*length(idx))),:);% 10% test
    yyTest=yyTrain(idx(1:ceil(0.1*length(idx))));
    
    xxTrain=xxTrain(idx(ceil(0.1*length(idx))+1:end),:);% 10% test
    yyTrain=yyTrain(idx(ceil(0.1*length(idx))+1:end));
end

yyTrain(yyTrain==2)=-1;% the label is 1 and -1
yyTest(yyTest==2)=-1;% the label is 1 and -1

xxTrain=full(xxTrain);
xxTest=full(xxTest);


% use these parameters if you dont want to select it.
optimalGamma=0.1/(2*mean(std(xxTrain).^2));
optimalLambda=1/length(yyTrain);

myflag=1; % regression
scheme_type='Removal'; %'Removal' or 'Projection
mybeta=0.5;
loss_type='Hinge'; % 'Hinge', 'Logistic'


%% Cross Validation to select parameters
if 0
    [ optimalLambda, optimalGamma ] = FindingParameters_NBSGD( xxTrain, yyTrain, 'scheme_type',scheme_type,...
        'loss_type',loss_type, 'beta',mybeta,'flag',myflag );
end

%% Training

% it would be faster if we set IsPlotLoss=0
tic;
[ output ] = NonparametricBudgeted_SGD(xxTrain,yyTrain,'scheme_type',scheme_type,'loss_type',loss_type,...
    'gamma',optimalGamma,'lambda',optimalLambda,'beta',mybeta,'IsPlotLoss',1,'flag',myflag);
trainTime=toc;

%% Plot
if isfield(output, 'myloss')
    figure;
    % loss wrt iteration
    plot(output.myloss,'-ms');
    ylabel('Loss');
    xlabel('Iteration');
    strTick=[500:500:output.TT];
    
    set(gca,'XTickLabel',(strTick))
    set(gca,'fontsize',14);
    title('Convergence of NBSGD','fontsize',18);
end

%% prediction
[ scores ] = NBSGDPrediction( output,xxTest,yyTest,optimalGamma,myflag);
scores.Time=trainTime;

if isfield(scores, 'F1')
    fprintf('F1 %.2f Acc %.2f Sparsity %.2f ModelSz %.2f Time %.1f (sec)\n',scores.F1,scores.Acc,scores.Spar,scores.ModSz,scores.Time);
else
    if isfield(scores,'RMSE')
        fprintf('RMSE %.2f MAE %.2f Sparsity %.2f ModelSz %.2f Time %.1f (sec)\n',scores.RMSE,scores.MAE,scores.Spar,scores.ModSz,scores.Time);
    else
        fprintf('Acc %.2f Sparsity %.2f ModelSz %.2f Time %.1f (sec)\n',scores.Acc,scores.Spar,scores.ModSz,scores.Time);
    end
end


