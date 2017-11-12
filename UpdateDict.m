function [D] = UpdateDict(Y, preD, X)%公式(15)中的Z矩阵，D中的第c类原子，X中的前C行
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
D       = preD;  %D中的第c类原子 [576*4]
nAtom   = size(D,2);  %求D得列数，普通的是4列,shared是5列 
Dindex  = [1:1:nAtom];  %1到4或者5 

for k = 1:nAtom
    idx = find(Dindex~=k);  %一类中的除了第k个原子
    if (norm([X(k,:)],'fro')<1E-4)  %对X(k,:)的转置*X(k,:)的迹开根号，[6*1]*[1*6]是[6*6]的方阵取迹，实际意义是X矩阵中的对应C类标签的某一行能量过小
                                    %其实就是ΣK(k.:).^2                         
        
%1、如果A为矩阵
% n=norm(A) 返回A的最大奇异值，即max(svd(A))
%n=norm(A,p)
%根据p的不同，返回不同的值
%p  返回值 
%1  返回A中最大一列和，即max(sum(abs(A))) 
%2  返回A的最大奇异值，和n=norm(A)用法一样 
%inf  返回A中最大一行和，即max(sum(abs(A’))) 
%‘fro’  A’和A的积的对角线和的平方根，即sqrt(sum(diag(A'*A)))
%2、如果A为向量
%norm(A,p)
%返回向量A的p范数。即返回 sum(abs(A).^p)^(1/p),对任意 1<p<+∞.
%norm(A)
%返回向量A的2范数，即等价于norm(A,2)。
%norm(A,inf)
%返回max(abs(A))
%norm(A,-inf)
%返回min(abs(A))
        E          = Y - D * X;  %Y是公式（23）中的残差矩阵，[576*4]*[4*6]=[576*6]
        E          = sum(E.^2);  %求E的列平方和的值
        [~, index] = max(E);  %取最大值所在的列,也就是样本误差最大的列
        D(:,k)     = E(:,index) ;  %D矩阵第k个原子换成E中的原子将D中该列换成最大残差列
        D(:,k)     = D(:,k) ./ norm(D(:,k));  %D的F范数归一
    else
        D(:,k) = (Y - D(:,idx)*X(idx,:)) * X(k,:)';  %公式（16）
        D(:,k) = D(:,k) ./ norm(D(:,k));  %D的F范数归一
    end
end
end