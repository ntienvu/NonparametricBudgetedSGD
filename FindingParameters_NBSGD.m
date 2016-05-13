function [ optimalLambda, optimalGamma ] = FindingParameters_NBSGD( xxTrain, yyTrain, varargin )
% FindingParameters_NBSGD performs cross validation to select the optimal
% parameters for Lambda and Gamma

[loss_type, scheme_type,beta,flag] = process_options(varargin,'loss_type',1,'scheme_type',1,'beta',0.5,'flag',1);

idx=randperm(length(yyTrain));
selxxTrain=xxTrain(idx(1:0.8*length(yyTrain)),:);
selyyTrain=yyTrain(idx(1:0.8*length(yyTrain)));
from=0.8*length(yyTrain)+1;
to=1*length(yyTrain);
selxxTest=xxTrain(idx(from:to),:);
selyyTest=yyTrain(idx(from:to));

gammaSet=[0.01 0.1 1 10 50];
lambdaSet=[100 5000 10000 100000]/length(yyTrain);
disp('================= Cross Validation =====================');

RMSESet=zeros(length(gammaSet),length(lambdaSet));
AccSet=zeros(length(gammaSet),length(lambdaSet));
for ii=1:length(gammaSet)
    for jj=1:length(lambdaSet)
        output = NonparametricBudgeted_SGD(selxxTrain,selyyTrain,'scheme_type',scheme_type,'loss_type',loss_type,...
           'IsPlotLoss',0, 'gamma',gammaSet(ii),'lambda',lambdaSet(jj),'beta',beta,'flag',flag);
        
        %evaluation
        [ scores ] = NBSGDPrediction( output,selxxTest,selyyTest,gammaSet(ii), flag );
        
        fprintf('Gam %.3f Lamb %.3f ',gammaSet(ii),lambdaSet(jj));
        if isfield(scores, 'F1')
            AccSet(ii,jj)=scores.Acc;
            fprintf('F1 %.2f Accuracy %.2f Sparsity %.2f ModelSz %d\n',scores.F1,scores.Acc,scores.Spar,scores.ModSz);
        else
            if isfield(scores,'RMSE')
                RMSESet(ii,jj)=scores.RMSE;
                fprintf('RMSE %.2f MAE %.2f Sparsity %.2f ModelSz %d\n',scores.RMSE,scores.MAE,scores.Spar,scores.ModSz);
            else
                AccSet(ii,jj)=scores.Acc;
                fprintf('Accuracy %.2f Sparsity %.2f ModelSz %d\n',scores.Acc,scores.Spar,scores.ModSz);
            end
        end
    end
end

% select optimal parameter
if isfield(scores, 'F1')
    AccSet(isnan(AccSet))=999999;
    AccSet(isinf(AccSet))=999999;
    [iRow, iCol]=find(AccSet==max(AccSet(:)));
else
    if isfield(scores,'RMSE')
        RMSESet(isnan(RMSESet))=999999;
        RMSESet(isinf(RMSESet))=999999;
        [iRow, iCol]=find(RMSESet==min(RMSESet(:)));
    else
        AccSet(isnan(AccSet))=999999;
        AccSet(isinf(AccSet))=999999;
        [iRow, iCol]=find(AccSet==max(AccSet(:)));
    end
end

optimalLambda=lambdaSet(iCol(1));
optimalGamma=gammaSet(iRow(1));
fprintf('Optimal Gam=%.3f Lamb=%.3f\n',optimalGamma,optimalLambda);
end

