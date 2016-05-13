function [ output ] = NonparametricBudgeted_SGD(xxTrain,yyTrain,varargin)
%% Nonparametric Budgeted SGD
%   input ===================================================================
%   xxTrain: feature of training data [NTrain x DD]
%   yyTrain: label of trainning data [NTrain x 1]
%   xxTest: feature of testing data [NTest x DD]
%   'gamma': the default value is 1/NTrain
%   'sigma': the default value is 1
%   'lambda': regularizer parameter, the default value is 32/NTrain
%   'beta': control the maintenance rate, the default value is NTrain*0.9
%   'BB': init budget size. the default value is 50
%   'flag': 1: binary classification 2: multiclassification 3: regression
%   'scheme_type': 1: removal 2: projection
%   'loss_type': 'Hinge' 'L2' 'L1' 'Logistic' 'Eps'

%   =========================================================================
%   output ==================================================================
%   output.alpha: coefficient parameters. [1 x ModelSize]
%   output.SupportIndex: indexing to the inducing points. [1 x ModelSize]
%   output.SupportVectors: a collection of inducing points. [ModelSize x DD]
%   Contact: ntienvu@gmail.com

warning off;
% check the data, then decide it is regression, binary classification or
% multiclassification
output=[];

[loss_type, scheme_type, BB,gamma,lambda,beta,IsPlotLoss,flag] = process_options(varargin,'loss_type',1,...
    'scheme_type',1,'BB',ceil(0.03*length(yyTrain)),'gamma',1/size(xxTrain,2),'lambda',32/length(yyTrain),...
    'beta',0.5,'IsPlotLoss',1,'flag',1);
    
switch flag
    case 1 % binary classification
        [ output ] = NBSGD_Binary_Regression(xxTrain,yyTrain,varargin);
    case 2 % multiclassification
        [ output ] = NBSGD_MultiClassification(xxTrain,yyTrain,varargin);
    case 3 % regression
        [ output ] = NBSGD_Binary_Regression(xxTrain,yyTrain,varargin);
        
        
end
end

function [ output ] = NBSGD_MultiClassification(xxTrain,yyTrain,varargin)
% perform MultiClassification

[loss_type, scheme_type, BB,gamma,lambda,beta,IsPlotLoss] = process_options(varargin{1},'loss_type',1,...
    'scheme_type',1,'BB',ceil(0.1*length(yyTrain)),'gamma',1/size(xxTrain,2),'lambda',32/length(yyTrain),'beta',0.5,'IsPlotLoss',1);

fprintf('%s %s\t',scheme_type,loss_type);


TT=length(yyTrain);
%yyTrain=uint8(categorical(yyTrain));

beta=beta*TT;

% permute the data
%idx=randperm(TT);
%yyTrain=yyTrain(idx);
%xxTrain=xxTrain(idx,:);

MaintenanceCount=0;

yyTrain=uint8(categorical(yyTrain));
KK = length(unique(yyTrain)); % Number of classes

% initialization
SupportIndex=[1];
alpha=zeros(1,KK); %matrix size(SupportIndex) x K

for tt=2:TT
    
    selected_idx=tt-1;
    xx = xxTrain(selected_idx,:);
    yy = yyTrain(selected_idx);
    
    
    % update w_t+1 from f'(w_t)
    alpha=alpha-alpha/tt;
    
    % compute first derivative of loss function
    % ywx
    % compute kernel K < xx, xx(SupportIndex) >
    
    KK_xt_xB = exp(-gamma.*pdist2(xxTrain(SupportIndex,:),xx,'euclidean').^2);
    
    term=KK_xt_xB'*alpha;
    
    left_term=term(yy);
    
    term(yy)=-999999;
    [right_term, max_not_yy]=max(term);
    aa=left_term-right_term;
    
    IsSatisfied=1;
    
    switch loss_type
        case 'Hinge'% Hinge
            if aa<1
                l_derv=-1;
            else
                l_derv=0;
                IsSatisfied=0;
            end
        case 'Logistic' % Logistic
            llaa=SigmoidFunction(-aa);
            l_derv=llaa*(llaa-1);
    end
    
    if IsSatisfied==1 % first derivative not zero
        eta=1/(lambda*tt);
        
        isemp=ismembc(SupportIndex,selected_idx);
        
        if sum(isemp)==0 % not in the Support Index
            alpha=[alpha; zeros(1,KK)];
            alpha(end,yy)=alpha(end,yy) -eta*l_derv;
            alpha(end,max_not_yy)=alpha(end,max_not_yy) +eta*l_derv;
            SupportIndex=[SupportIndex selected_idx];
        else
            tempidx=find(SupportIndex==selected_idx);
            alpha(tempidx,yy)=alpha(tempidx,yy) -eta*l_derv;
            alpha(tempidx,max_not_yy)=alpha(tempidx,max_not_yy) +eta*l_derv;
        end
        %w_t+1 = w_t+1 - eta_t l'()
        
        b=size(alpha,1);
        if b>BB % budget maintenance
            MaintenanceCount=MaintenanceCount+1;
            
            %random from Bernoulli
            st=min(beta/tt,1);
            uu = rand;
            ber = sum(uu>cumsum([(1-st) st ]));
            
            if ber==0
                BB=BB+1;
                continue;
            end
            
            [~, p_idx]=min(sqrt(sum(alpha.^2,2)));
            MinVectors=xxTrain(SupportIndex(p_idx),:);
            
            if scheme_type==1                 % Removal
                alpha(p_idx,:)=[];
                SupportIndex(p_idx)=[];
            end
            
            if scheme_type==2                   % Projection
                
                if size(alpha,1)>5
                    top5=[1:5];
                else
                    top5=[1:size(alpha,1)];
                end
                    
                VecTop5=xxTrain(SupportIndex(top5),:);
                KK_top5_top5=exp(-gamma.*pdist2(VecTop5,xxTrain(SupportIndex(top5),:),'euclidean').^2);
                
                KK_inv=pinv(KK_top5_top5);
                
                KK_xpidx_xB = exp(-gamma.*pdist2(xxTrain(SupportIndex(top5),:),MinVectors,'euclidean').^2);
                
                delta_alpha=KK_inv*KK_xpidx_xB*alpha(p_idx,:);
               
                alpha(top5,:)=alpha(top5,:)+delta_alpha;
                
                alpha(p_idx,:)=[];
                SupportIndex(p_idx)=[];
            end
        else
            
        end
    end
    
end

output.alpha=alpha;
output.xx=xxTrain(SupportIndex,:);
output.yy=yyTrain(SupportIndex);
output.SupportIndex=SupportIndex;
output.MaintenanceCount=MaintenanceCount;
output.SupportVectors=xxTrain(SupportIndex,:);
output.TT=TT;
end


function [ output ] = NBSGD_Binary_Regression(xxTrain,yyTrain,varargin)
% perform Binary Classification vs Regression

MaintenanceCount=0;

warning off;
[loss_type, scheme_type, IsPlotLoss,BB,gamma,lambda,beta] = process_options(varargin{1},'loss_type',1,'scheme_type',1,...
    'IsPlotLoss',0,'BB',ceil(0.1*length(yyTrain)),'gamma',1/size(xxTrain,2),'lambda',32/length(yyTrain),'beta',0.5);

fprintf('%s %s\t',scheme_type,loss_type);


epsilon=0.1; % for epsilon insensitive loss

yyTrain=double(yyTrain);
TT=length(yyTrain);

beta=beta*TT;

dim=size(xxTrain,2);

% permute the data
idx=randperm(TT);
yyTrain=yyTrain(idx);
xxTrain=xxTrain(idx,:);

SupportIndex=[1];
alpha=0;

startTime=tic;

myloss=[];
mytime=[];
TimeArray=[];
ModelSzArray=[];
for tt=2:TT
    %alpha=double(alpha);
    if IsPlotLoss==1 && mod(tt,500)==0
        % compute accumulate loss
        KK = exp(-gamma.*pdist2(xxTrain(SupportIndex,:),xxTrain,'euclidean').^2);
        
        KK_support=exp(-gamma.*pdist2(xxTrain(SupportIndex,:),xxTrain(SupportIndex,:),'euclidean').^2);
        
        regularizer=0.5.*alpha*KK_support*alpha';
        decision=(yyTrain').*(alpha*KK);
        tempVal=1-decision;
        tempVal(tempVal<0)=0;
        myloss=[myloss mean(tempVal)+regularizer];
        
        temp=toc(startTime);
        mytime=[mytime temp];
        
        TimeArray=[TimeArray temp];
        ModelSzArray=[ModelSzArray length(SupportIndex)];
        
        output.myloss=myloss;
        output.mytime=mytime;
        
    end
    
    
    selected_idx=mod(tt,TT)+1;
    
    xx = xxTrain(selected_idx,:);
    yy = yyTrain(selected_idx);
    
    % update w_t+1 from f'(w_t)
    alpha=alpha-alpha/tt;
    
    % compute first derivative of loss function
    % ywx
    % compute kernel K < xx, xx(SupportIndex) >
    KK_xt_xB = exp(-gamma.*pdist2(xxTrain(SupportIndex,:),xx,'euclidean').^2);
    
    IsSatisfied=1;
    appended_val=0;
    
    switch loss_type
        case 'Hinge'% Hingle
            temp=full(sum(yy*alpha*KK_xt_xB));% hingle loss
            if temp>=1
                IsSatisfied=0;
            end
            appended_val=0-yy;
        case 'L2'% L2
            temp=full(sum(alpha*KK_xt_xB));
            if temp==yy
                IsSatisfied=0;
            end
            appended_val=(temp-yy);
        case 'L1' % L1
            temp=full(sum(alpha*KK_xt_xB));
            if temp==yy
                IsSatisfied=0;
            end
            appended_val=sign(temp-yy);
        case 'Logistic' % Logistic
            temp=full(sum(-yy*alpha*KK_xt_xB));
            appended_val=-yy*exp(temp)/(exp(temp)+1);
        case 'Eps' % Epsilon
            temp=full(sum(alpha*KK_xt_xB))-yy;
            if (abs(temp)<=epsilon) && (temp==0)
                IsSatisfied=0;
            end
            appended_val=sign(temp);
    end
    
    appended_val=double(appended_val);
    if IsSatisfied==1 % first derivative not zero
        eta=1/(lambda*tt);
        
        isemp=sum(ismembc(SupportIndex,selected_idx));
        if isemp==0 % not in the Support Index
            alpha=[alpha, -eta*appended_val];% = [alpha, -1/tt];
            SupportIndex=[SupportIndex selected_idx];
        else
            tempidx=find(SupportIndex==selected_idx);
            alpha(tempidx)=alpha(tempidx)-eta*appended_val;
        end
        %w_t+1 = w_t+1 - eta_t l'()
        
        b=length(alpha);
        
        if b>BB % budget maintenance
            
            %random from Bernoulli
            st=min(beta/tt,1);
            uu = rand;
            ber = sum(uu>cumsum([(1-st) st]));
            
            if ber==0
                BB=BB+1;
                continue;
            end
            
            MaintenanceCount=MaintenanceCount+1;
            [~, p_idx]=min(abs(alpha));
            MinVectors=xxTrain(SupportIndex(p_idx),:);
            
            if strcmp(scheme_type,'Removal')                % scheme 1: Removal
                alpha(p_idx)=[];
                SupportIndex(p_idx)=[];
            end
            
            if strcmp(scheme_type,'Projection')            % scheme 2: Projection
                
                if length(alpha)>5
                    top5=[1:5];
                else
                    top5=[1:length(alpha)];
                end
                VecTop5=xxTrain(SupportIndex(top5),:);
                KK_top5_top5=exp(-gamma.*pdist2(VecTop5,VecTop5,'euclidean').^2);
                
                KK_inv=pinv(KK_top5_top5);
                
                KK_xpidx_xB = exp(-gamma.*pdist2(VecTop5,MinVectors,'euclidean').^2);
                
                delta_alpha=alpha(p_idx)*KK_inv*KK_xpidx_xB;

                
                alpha(top5)=alpha(top5)+delta_alpha';
                
                alpha(p_idx)=[];
                SupportIndex(p_idx)=[];
            end
        else
        end
    end
end

output.alpha=alpha;
output.xx=xxTrain(SupportIndex,:);
output.yy=yyTrain(SupportIndex);
output.SupportIndex=SupportIndex;
output.MaintenanceCount=MaintenanceCount;
output.SupportVectors=xxTrain(SupportIndex,:);
output.TT=TT;
end
