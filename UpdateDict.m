function [D] = UpdateDict(Y, preD, X)%��ʽ(15)�е�Z����D�еĵ�c��ԭ�ӣ�X�е�ǰC��
% Function: updata the dictionary atom by atom
% D_hat = argmin||Y - D * X||^2, 
% Input:
%     Y      -- the training data, each column signifies one sample
%
%     preD   -- the dictionary to be updated 
% 
%     X      -- the representations over D for the samples X
%
%     maxiter -- the maximum iteration number
%
% Output:
%      D      -- the updated dictionary
%  Written by Xiudong Wang (wang-xd14@mails.tsinghua.edu.cn)
%  date 5/1/2017
%%
D       = preD;  %D�еĵ�c��ԭ�� [576*4]
nAtom   = size(D,2);  %��D����������ͨ����4��,shared��5�� 
Dindex  = [1:1:nAtom];  %1��4����5 

for k = 1:nAtom
    idx = find(Dindex~=k);  %һ���еĳ��˵�k��ԭ��
    if (norm([X(k,:)],'fro')<1E-4)  %��X(k,:)��ת��*X(k,:)�ļ������ţ�[6*1]*[1*6]��[6*6]�ķ���ȡ����ʵ��������X�����еĶ�ӦC���ǩ��ĳһ��������С
                                    %��ʵ���Ǧ�K(k.:).^2                         
        
%1�����AΪ����
% n=norm(A) ����A���������ֵ����max(svd(A))
%n=norm(A,p)
%����p�Ĳ�ͬ�����ز�ͬ��ֵ
%p  ����ֵ 
%1  ����A�����һ�кͣ���max(sum(abs(A))) 
%2  ����A���������ֵ����n=norm(A)�÷�һ�� 
%inf  ����A�����һ�кͣ���max(sum(abs(A��))) 
%��fro��  A����A�Ļ��ĶԽ��ߺ͵�ƽ��������sqrt(sum(diag(A'*A)))
%2�����AΪ����
%norm(A,p)
%��������A��p������������ sum(abs(A).^p)^(1/p),������ 1<p<+��.
%norm(A)
%��������A��2���������ȼ���norm(A,2)��
%norm(A,inf)
%����max(abs(A))
%norm(A,-inf)
%����min(abs(A))
        E          = Y - D * X;  %Y�ǹ�ʽ��23���еĲв����[576*4]*[4*6]=[576*6]
        E          = sum(E.^2);  %��E����ƽ���͵�ֵ
        [~, index] = max(E);  %ȡ���ֵ���ڵ���,Ҳ�����������������
        D(:,k)     = E(:,index) ;  %D�����k��ԭ�ӻ���E�е�ԭ�ӽ�D�и��л������в���
        D(:,k)     = D(:,k) ./ norm(D(:,k));  %D��F������һ
    else
        D(:,k) = (Y - D(:,idx)*X(idx,:)) * X(k,:)';  %��ʽ��16��
        D(:,k) = D(:,k) ./ norm(D(:,k));  %D��F������һ
    end
end
end