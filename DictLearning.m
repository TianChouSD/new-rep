function [D, X] = DictLearning(DInit, DLabel, XInit, trDat, trls, C, beta, lambda, gamma, MAXITER)
%% 
%  Function: learn the dictionary and the represeantation the with the training samples
%
%  Inputs:
%
%    DInit  -- the initial dictionary ��ʼ���ֵ����
%   
%    DLabel -- the label table for the whole dictionary �ֵ����ԭ����ӳ������һ���
%
%    XInit  -- the initial representation ��ʼ����X����
%
%    trDat  -- the training samples with each column denoting one sample һ��15��*6��/���ѵ������
%    
%    trls   -- the ground truth for the training set 'tr_dat'  ԭ��165��������ѡ�е���
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
D       = DInit; %��ʼ���ֵ����
X       = XInit; %��ʼ������X����
while iter <= MAXITER   %���������� ��iteration�� ����
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %  update coefficients   %
    %%%%%%%%%%%%%%%%%%%%%%%%%% 

    fprintf('\n%d/%d\tupdate the representations......', iter, MAXITER);
    for c = 1:C
        fprintf('.');
        Y           = trDat(:,trls == c);  %ÿ��ѡ��6������
        index_cIdv  = find(DLabel == c);  %��C���ǩ��Ӧ��D����������4��λ��
        index_Share = find(DLabel == C+1);  %����ԭ����DLabel�е�λ��
        outIndex    = setdiff(1:size(D,2), [index_cIdv index_Share]);  %ѡ���ǵ�C���ҷǹ�����ԭ��

        nTrain = size(Y,2);   %��������6�����ǳ����6����������ʽ(8)�е�NC
        weight = 1/(nTrain-1)*[ones(nTrain,nTrain)-eye(nTrain)];   %L^��C�����P13ҳ��ʽ(8)
        lap    = diag(sum(weight))-weight;  %��13ҳ�Ĺ�ʽ(8),lap�����L�ϱ�c����
        
        Pc     = zeros(size(outIndex,2),size(D,2));  %����һ����65-5-4��*65��ȫ0��
        Pc(:,outIndex) = eye(size(outIndex,2));  %Pc=[0 eye(65-5-4) 0]������c=1,ǰ4��Ϊ0����5��Ϊ0���м�и���λ��
        Y_hat  = D'*Y-gamma*X{c}*(lap-diag(diag(lap)));  %X{c}*(lap-diag(diag(lap)))�õ�65*6��ϡ��X�󣬶�Ӧ��c��
%         X{c}   = ( D'*D + lambda*(Pc'*Pc) + (gamma + beta)*eye(size(D,2)) )\Y_hat;  %������һ�ж�Ӧ�ڹ�ʽ(13)����ʱX����������
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

    Scode       = cell2mat(X); %1*15��65*6��ÿ��X����mat����65*(16*5)�ľ���
    nLabelAtom  = C; %һ��15��
    shDSize     = sum(DLabel==C+1); %����ԭ�ӵĸ���
    if(shDSize > 0)
        nLabelAtom = C+1; % shared atom to be updated also���빫��ԭ��
    end
    % update label-particular and shared atoms
    for c=1:1:nLabelAtom
        fprintf('.');
        if(c<=C && lambda >= 2E1 ) 
            %when lambda is very large, for samples from other classes except the c^th class
            %the coefficients located at the c^th class atoms is little. 
            Y     = trDat(:,trls==c); %ѡ�������еĵ�c���6��
            temS  = X{c}; %ѡ����ʼ�����X���Ӧ�ڵ�c��ľ���
        else
            Y     = trDat; %����ԭ�Ӷ�Ӧ����������������
            temS  = Scode; %����ԭ�Ӷ�Ӧ������X����
        end
        index         = DLabel == c; %Dlabel����c���ȡ1������Ϊ0
        indexOut      = ~index; %��index
        YRes          = Y - D(:,indexOut) * temS(indexOut,:); %��ʽ(15)�е�Z����ѵ������������޳��˸����ڵı�ǩ�������У�temS(indexOut,:)�Ǽ���ȫ0�ģ�YRes��Y
        D(:,index)    = UpdateDict(YRes, D(:,index), temS(index,:)); %����D����ĵ�c��ԭ��
    end   
    iter = iter+1; %��������+1�������ǵ���15��
end
end