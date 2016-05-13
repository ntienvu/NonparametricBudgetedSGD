function [ scores ] = NBSGDPrediction( model,xxTest,yyTest,gamma, flag )
% given the model and testing data
% compute perfomance scores
% scores: F1, Acc for classification
% scores: RMSE, MAE for regression

TT=model.TT; % to compute the sparsity
SupportVectors=model.SupportVectors;
KK = exp(-gamma.*pdist2(SupportVectors,xxTest,'euclidean').^2);

switch flag
    case 1 % binary classification
    
        
        decision=model.alpha*KK;

        predLabel=ones(1,length(yyTest));
        predLabel(decision<0)=-1;

        yyTest=uint8(categorical(yyTest));
        predLabel=uint8(categorical(predLabel));

        % compute Accuracy
        Acc=length(find(yyTest==predLabel'))/length(predLabel);
    
        % compute F1 score
        [F1, rec, spec, prec] = F1_Evaluation(yyTest-1, predLabel'-1); % convert from 1,2 to 0,1
        %fprintf(' Acc F1 Time Sparsity ModSize %.2f %.2f %.2f %.2f %d\n',Acc,F1,trainTime+testTime,...
            %length(output.alpha)/TT,length(output.alpha));
            
        scores.Acc=Acc;
        scores.F1=F1;
        scores.Spar=length(model.alpha)/TT;
        scores.ModSz=length(model.alpha);

    case 2 % multi-classification
        
        decision=KK'*model.alpha;% it is slight different from binary and regression cases
 
        [~, predLabel]=max(decision');
        
        yyTest=uint8(categorical(yyTest));
        predLabel=uint8(categorical(predLabel));
        
        Acc=length(find(yyTest==predLabel'))/length(predLabel);
    
        %fprintf(' Acc Time Sparsity ModSize %.2f %.2f %.2f %.2f %d\n',Acc,trainTime+testTime,...
            %length(output.alpha)/TT,length(output.alpha));
            
        scores.Acc=Acc;
        scores.Spar=length(model.alpha)/TT;
        scores.ModSz=length(model.alpha);
        
    case 3 % regression
    
        decision=double(model.alpha*KK);
        yyTest=double(yyTest);

        RMSE=sqrt(mean((decision'-yyTest(1:length(decision))).^2));
        MAE=mean(abs(decision'-yyTest(1:length(decision))));
        %fprintf('\tTime [Train=%.2f Test=%.2f Total=%.2f] \t nMtn=%d RMSE= %.2f MAE=%.2f Sparsity=%.2f ModSz=%d\n',trainTime,testTime,...
            %trainTime+testTime,output.MaintenanceCount,RMSE,MAE,length(output.alpha)/TT,length(output.alpha));

        scores.RMSE=RMSE;
        scores.MAE=MAE;
        scores.Spar=length(model.alpha)/TT;
        scores.ModSz=length(model.alpha);
        %fprintf(' RMSE MAE Time Sparsity ModSize %.2f %.2f %.2f %.2f %d\n',RMSE,MAE,trainTime+testTime,length(output.alpha)/TT,length(output.alpha));
end

end

