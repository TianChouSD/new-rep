function [X] = InitCoeff(Dict, trDat, trLab, C, beta) %��ʼ����D���󣬳�ȡ��576*90���������������ж�Ӧ����������15�࣬����beta
%% 
%  Function: initialize the representation for training samples
%
%  Inputs:
%
%    Dict   -- a structured dictionary with each column denoting one labeled atom
%   
%    trDat -- the training samples with each column denoting one sample
%
%    trls   -- the ground truth for the training set 'trDat'
%
%    C      --the total number of classes
%
%    beta   -- a scalar for the coding regularization
%
%  Output:
%
%    X -- the initilized representation 
%
%  Written by Xiudong Wang (wang-xd14@mails.tsinghua.edu.cn)
%  date 5/1/2017
%%
X          = cell(1,C);
for c = 1:C 
    Y      = trDat(:,trLab == c); %�ѵ�c��������ҳ�����ÿ��6������
    X{c}   = ( Dict'*Dict + beta*eye(size(Dict,2)) )\(Dict'*Y); %'a\b'����a����*b��ʵ������([65*576]*[576*65]+betaeye(65))����*[65*576]*[576*6]
end

end