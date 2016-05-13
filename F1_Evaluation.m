function [F1, rec, spec, prec] = F1_Evaluation(true_label, pred_label, varargin)
%TUND_ROC Summary of this function goes here
%   Detailed explanation goes here

% %% parse input arguments
% parser = inputParser;
% % parser.addRequired('true_label', @isnumeric || @islogical);
% % parser.addRequired('predicted_label', @isnumeric || @islogical);
% 
% parser.parse(true_label, predicted_label, varargin{:});

% check the label -1 and 1, then convert to 0 and 1
temp=unique(true_label);
if isequal(temp,[-1 1])
    true_label=uint8(categorical(true_label))-1;
    pred_label=uint8(categorical(pred_label))-1;
end

%%
TP = full(sum(true_label & pred_label));
FN = full(sum(true_label & (~pred_label)));
FP = full(sum((~true_label) & pred_label));
TN = full(sum((~true_label) & (~pred_label)));

rec = TP / (TP+FN+realmin);
prec = TP / (TP+FP+realmin);
F1 = 2*(rec*prec) / (rec+prec+realmin);
spec = TN / (FP+TN+realmin);

end

