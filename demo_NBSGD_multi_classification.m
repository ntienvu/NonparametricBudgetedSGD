clear all;
addpath('release_NBSGD');

% There are 5 losses: Hinge, L2, L1, Logistic, Eps-Intensitive
% User can select loss_type=1,2,3,4,5 for the above losses.
% Hingle and Logistic losses are for classification
% L2, L1 and Eps-Intensitive losses are for regression


% binary classification
fprintf('MultiClass ');


[yyTrain, xxTrain]=libsvmread('F:\Data\multiclassification\dna.scale.tr');[yyTest, xxTest]=libsvmread('F:\Data\multiclassification\dna.scale.t');
    %[yyTrain, xxTrain]=libsvmread('F:\Data\multiclassification\letter.scale.tr');[yyTest, xxTest]=libsvmread('F:\Data\multiclassification\letter.scale.t');


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

xxTrain=full(xxTrain);
xxTest=full(xxTest);

myflag=2; % myflag=2 indicates multi classification task

% use these parameters if you dont want to select it.
optimalGamma=0.1/(2*mean(std(xxTrain).^2));
optimalLambda=1/length(yyTrain);

scheme_type='Removal'; %'Removal' or 'Projection
mybeta=0.3;
loss_type='Logistic'; % 'Hinge', 'Logistic'


%% Cross Validation to select parameters
if 0
    [ optimalLambda, optimalGamma ] = FindingParameters_NBSGD( xxTrain, yyTrain, 'scheme_type',scheme_type,...
        'loss_type',loss_type, 'beta',mybeta,'flag',myflag );
end

%% Training

tic;
[ output ] = NonparametricBudgeted_SGD(xxTrain,yyTrain,'scheme_type',scheme_type,'loss_type',loss_type,...
    'gamma',optimalGamma,'lambda',optimalLambda,'beta',mybeta,'flag',myflag);
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
    fprintf('F1 %.2f Accuracy %.2f Sparsity %.2f ModelSz %.2f Time %.1f (sec)\n',scores.F1,scores.Acc,scores.Spar,scores.ModSz,scores.Time);
else
    if isfield(scores,'RMSE')
        fprintf('RMSE %.2f MAE %.2f Sparsity %.2f ModelSz %.2f Time %.1f (sec)\n',scores.RMSE,scores.MAE,scores.Spar,scores.ModSz,scores.Time);
    else
        fprintf('Accuracy %.2f Sparsity %.2f ModelSz %.2f Time %.1f (sec)\n',scores.Acc,scores.Spar,scores.ModSz,scores.Time);
    end
end


