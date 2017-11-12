function [D, X] = DictLearning(DInit, DLabel, XInit, trDat, trls, C, beta, lambda, gamma, MAXITER)
%% 
%  Function: learn the dictionary and the represeantation the with the training samples
%
%  Inputs:
%
%    DInit  -- the initial dictionary 初始化字典矩阵
%   
%    DLabel -- the label table for the whole dictionary 字典矩阵原子是映射于哪一类的
%
%    XInit  -- the initial representation 初始化的X矩阵
%
%    trDat  -- the training samples with each column denoting one sample 一共15类*6个/类的训练样本
%    
%    trls   -- the ground truth for the training set 'tr_dat'  原来165列样本中选中的列
%
%    C      -- the total number of classes
%
%    beta   -- a scalar for the L2-norm coding regularization
%
%    lambda -- a scalar of the cross-label suppression
%
%    gamma  -- a scalar of the group regularization
%
%  Output:
%
%      D     -- the learnt dictionary 
%
%      X     -- the learnt representation 
%
%  Written by Xiudong Wang (wang-xd14@mails.tsinghua.edu.cn)
%  date 5/1/2017
%%
iter    = 1;
D       = DInit; %初始化字典矩阵
X       = XInit; %初始化特征X矩阵
while iter <= MAXITER   %最大迭代次数 ‘iteration’ 迭代
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %  update coefficients   %
    %%%%%%%%%%%%%%%%%%%%%%%%%% 

    fprintf('\n%d/%d\tupdate the representations......', iter, MAXITER);
    for c = 1:C
        fprintf('.');
        Y           = trDat(:,trls == c);  %每次选出6列样本
        index_cIdv  = find(DLabel == c);  %第C类标签对应的D矩阵连续的4个位置
        index_Share = find(DLabel == C+1);  %公共原子在DLabel中的位置
        outIndex    = setdiff(1:size(D,2), [index_cIdv index_Share]);  %选出非第C类且非公共的原子

        nTrain = size(Y,2);   %列数就是6，就是抽出的6个样本，公式(8)中的NC
        weight = 1/(nTrain-1)*[ones(nTrain,nTrain)-eye(nTrain)];   %L^（C）详见P13页公式(8)
        lap    = diag(sum(weight))-weight;  %见13页的公式(8),lap阵就是L上标c矩阵
        
        Pc     = zeros(size(outIndex,2),size(D,2));  %制造一个（65-5-4）*65的全0阵？
        Pc(:,outIndex) = eye(size(outIndex,2));  %Pc=[0 eye(65-5-4) 0]，比如c=1,前4列为0，后5列为0，中间夹个单位阵
        Y_hat  = D'*Y-gamma*X{c}*(lap-diag(diag(lap)));  %X{c}*(lap-diag(diag(lap)))得到65*6的稀疏X阵，对应第c类
%         X{c}   = ( D'*D + lambda*(Pc'*Pc) + (gamma + beta)*eye(size(D,2)) )\Y_hat;  %和上面一行对应于公式(13)，此时X矩阵更新完毕
        iterating=0;
        while iterating<=5;
        A=(D'*(D*X{c}-trDat(find(trls==c)))+beta*X{c}+lambda*Pc'*Pc*X{c}+gamma*X{c}*lap);
        A=0.1*A*(norm(X{c},'fro'))^2/(norm(A,'fro'))^2;
        X{c}=X{c}-A; 
        iterating=iterating+1;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %   update dictionary    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('\n\t\tupdate the dictionary.....');

    Scode       = cell2mat(X); %1*15个65*6的每类X阵变成mat后是65*(16*5)的矩阵
    nLabelAtom  = C; %一共15类
    shDSize     = sum(DLabel==C+1); %公共原子的个数
    if(shDSize > 0)
        nLabelAtom = C+1; % shared atom to be updated also加入公共原子
    end
    % update label-particular and shared atoms
    for c=1:1:nLabelAtom
        fprintf('.');
        if(c<=C && lambda >= 2E1 ) 
            %when lambda is very large, for samples from other classes except the c^th class
            %the coefficients located at the c^th class atoms is little. 
            Y     = trDat(:,trls==c); %选出样本中的第c类的6列
            temS  = X{c}; %选出初始化后的X阵对应于第c类的矩阵
        else
            Y     = trDat; %公共原子对应于整个抽样的样本
            temS  = Scode; %公共原子对应的整个X矩阵
        end
        index         = DLabel == c; %Dlabel中是c类的取1，其余为0
        indexOut      = ~index; %非index
        YRes          = Y - D(:,indexOut) * temS(indexOut,:); %公式(15)中的Z矩阵，训练到最后由于剔除了该属于的标签的能量行，temS(indexOut,:)是几乎全0的，YRes≈Y
        D(:,index)    = UpdateDict(YRes, D(:,index), temS(index,:)); %更新D矩阵的第c个原子
    end   
    iter = iter+1; %迭代次数+1，这里是迭代15次
end
end