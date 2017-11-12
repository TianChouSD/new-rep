function [X] = InitCoeff(Dict, trDat, trLab, C, beta) %初始化的D矩阵，抽取的576*90的样本，抽样的列对应的类向量，15类，参数beta
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
    Y      = trDat(:,trLab == c); %把第c类的样本找出来，每类6个样本
    X{c}   = ( Dict'*Dict + beta*eye(size(Dict,2)) )\(Dict'*Y); %'a\b'代表a的逆*b，实质上是([65*576]*[576*65]+betaeye(65))的逆*[65*576]*[576*6]
end

end