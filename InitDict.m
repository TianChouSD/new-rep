function [Dict, DLabel] = InitDict(trDat, trLab, C, options)  %trdat是抽样的15类*6列/类的矩阵，trLab是对应于第C类抽取的tr_dat中的列数，C是类的总数，options是一组参数的结构体
%% 
%  Function: initialize the dictionary with training data
%
%  Inputs:
%   
%    trDat    -- the training samples with each column denoting one sample
%
%    trLab    -- the ground truth for the training set 'trDat'
%
%    C        -- the total number of classes
%
%    options   -- the options for dictionary initialization
%       partDsizeList -- the part-dictionary size list for each category                               
%             shDSize -- the number of the shared atoms
%             wayInit -- the initialization method, like kmeans, pca, ksvd, and sequence
%                beta -- the scalar for the l2-norm regularizatin
%               initT -- l0-norm threshold for part dictionary with ksvd initialization
%             initShT -- l0-norm threshold for shared atoms with ksvd initialization
%             maxIter -- the max iter num for the ksvd method
%  Output:
%
%    Dict       --  the initilized dictionary 
%
%    DLabel     --  the label table for the dictionary
%
%  Written by Xiudong Wang (wang-xd14@mails.tsinghua.edu.cn)
%  date 5/1/2017
%%
if (isfield(options,'pDSizeList'))
    pDSizeList = options.pDSizeList;
    if (size(pDSizeList,2)==1)
        pDSizeList = repmat(pDSizeList, [1,C]);%如果只有一列，就把这个数复制成1*C（15）的矩阵
    end
    if (size(pDSizeList,2) < C)
         error('The pDSizeList is not specified');
    end
else
    error('The pDSizeList is not specified');
end     %这段把pDSizeList这个东西复制成了[1,C]的矩阵

if (~isfield(options,'shDSize'))
    shDSize = 0;
else
    shDSize = options.shDSize;
end     %

if (~isfield(options,'wayInit'))
    wayInit = 'kmeans';
else
    wayInit = options.wayInit;
end

if (strcmpi(wayInit, 'ksvd'))
    if (~isfield(options,'initT'))
        error('initT for ksvd is not specified');
    end
    if (~isfield(options,'initShT') && shDSize>0)
        error('shT for ksvd is not specified');
    end
    if (~isfield(options,'maxIter'))
        maxIter = 20;
    else
        maxIter = options.maxIter;
    end
end            %这些总的来说是判断结构体的语句，除了复制矩阵的有数值运算外，其余的只是为了程序的健壮而写

indDictSize = sum(pDSizeList); %不就是15类*每一类4个原子
Dict        = zeros(size(trDat,1),indDictSize+shDSize); %初始化576*（15*4+5)的全0矩阵indDictSize是15*4，shDSize是公共原子个数
DLabel      = zeros(1, indDictSize + shDSize); %---------------只能说1*65的全0阵，目前没有更深层次的理解！！！！及时改啊！！！-------------------

%% initilize the dictionary
%initilize the whole dictionary with randomly selected samples
if(strcmpi(wayInit,'wholeRand')) %全部随机包括类原子和公共原子全部随机，kmwans是无监督的聚类算法，质心迭代至收敛，分类毕，很简单
    nData = size(trDat,2); %选取的样本90列
    nAtom = size(Dict,2); %D矩阵65列
    Dict  = trDat(:, randperm(nData, nAtom)); %在样本中随机乱序抽取65列并排序，完全随机,注意，已经归一了哦
    pos   = 1;
    for c = 1:C
         perDSize = pDSizeList(c); %放心吧，既然是repmat的，每一个都一样的
         DLabel(pos: pos+perDSize-1) = c; %一共60列，4个1,4个2，一直到4个15
         pos = pos + perDSize; %pos=pos+D矩阵对于每一类分配原子的列数，repmat之后都一样啊
    end
    DLabel(pos: pos+shDSize-1) = C+1; %公共原子放在最后
    Dict = normcols(Dict); %按列平方和归一，关键前面没用这个函数而是用公式归一的。。。醉了，是一个人写的吗？
    return;
end

%initialize each part-dictionaries 这段初始化的是对应每一类的原子，下面一段有初始化公共原子
pos = 1;
for c = 1:C
    partDsize = pDSizeList(c);  %每一类有pDSizeList个原子，放心吧，既然是repmat的，每一类都一样的
    fprintf('\tClass-%d''s part-dictionary initialization...\n',c); %每找出一类就显示一些字符
    col_ids  = find(trLab == c);  %样本中分类编码等于1-15的位置，就是90个样本的分类数据，在Quora上看到的用matrix代替循环的典型例子，
                                  %matlab执行for语句的“能力”差，原因之一是按列储存元素导致指针查找会增加空间复杂度
    % ensure no zero data elements are chosen
    data_ids = find(sum(trDat(:,col_ids).^2,1) > 1e-6);  %对应的从1到15类，进行列求和剔除无能量的样本 ensure no zero data elements are chosen
    % perm     = randperm(length(data_ids));
    if strcmpi(wayInit,'ksvd')  
        perm           = [1:length(data_ids)]; 
        Dpart          = trDat(:,col_ids(data_ids(perm(1:partDsize))));
        para.data      = trDat(:,col_ids(data_ids));
        para.Tdata     = options.initT;
        para.iternum   = maxIter;
        para.memusage  = 'high';
        para.initdict1 = normcols(Dpart);
        para.dictsize  = partDsize;
        [Dpart,~,~]    = ksvd(para,'');
    else
        Dpart = Dictionary_Ini(trDat(:,col_ids(data_ids)),partDsize,wayInit);  %第一个参数表示随机的样本选出的有能量的列样本矩阵，
                                                                               %第二个是每一类样本的个数，第三个是模式选择的句柄
    end
    Dict(:, pos:pos+partDsize-1)  = Dpart;  %对对应第C类的字典矩阵的原子进行赋值
    DLabel(pos : pos+partDsize-1) = c;  %对应的原子（原子就是列）进行标注第几类
    pos = pos+partDsize;  %自行理解，哈哈哈哈哈哈哈嗝
end %到此为止前面60列的分类原子

% initialize the shared atoms  初始化公共的原子
if( shDSize > 0)  %shDSize就是公共原子的个数
    fprintf('\tThe shared part initialization...\n');    
    
    % substract the individual component
    resData=trDat;  %复制训练样本的矩阵，就是每类随机挑选6列随机排序的那个
    for c = 1:C
        indData   = resData(:,trLab == c);  %选出第c类的原子
        partD     = Dict(:,DLabel == c);  %先复制D矩阵的第c个原子的
        partX     = (partD'*partD + options.beta * eye(size(partD,2)))\(partD'*indData); %shared atom公式
        resData(:,trLab == c) = indData - partD * partX; %同上的公式
    end
    
    data_ids = find(sum(resData.^2,1) > 1e-6);   % ensure no zero data elements are chosen
    if(length(data_ids)>1)
       if(strcmp(wayInit,'ksvd'))
            perm          = [1:length(data_ids)]; 
            Dpart         = resData(:,data_ids(perm(1:shDSize)));
            para.dictsize = shDSize;
            para.data     = resData(:,data_ids);
            para.Tdata    = options.initShT;
            para.iternum  = maxIter;
            para.memusage = 'high';
            para.initdict = normcols(Dpart);
            [shD,~,~]     = ksvd(para,'');
       else
            shD = Dictionary_Ini(resData(:,data_ids), shDSize, wayInit); %在上述的剔除全0列后赋值，只要没有全0原子，就不变
       end
    else
        shD = trDat(:,1:shDsize); %？？？如果非全0列只有一行或者以下，就赋值前5行？？？
    end
    Dict(:, end+1-shDSize:end) = shD; 
    DLabel(end+1-shDSize:end)  = C+1;
end
end

function D    =    Dictionary_Ini(data, nCol, wayInit) %第一个参数是挑选出的非零样本矩阵，第二个是每一类样本的个数，第三个是模式选择的句柄
%  function: initialize dictionary
%   
%  Inputs
%   
%     data    -- data matrix. Each column vector of data denotes a sample 
%                 
%     nCol    -- the number of Dict's columns, i.e. the dictionary size
%
%     wayInit -- the initial method 
%
%   Outputs
%
%      D      -- initilazed dictionary
%%
m   =    size(data,1); %样本矩阵的行数，每个原子有多少个提取的特征
switch lower(wayInit)
    case {'pca'}  %数模里遇到过，是主成分分析
        [D,disc_value,Mean_Image]   =    Eigenface_f(data,nCol-1);
         D                          =    [D Mean_Image];
    case {'random'} %随机
         D                          =    randn(m, nCol);
    case {'kmeans'} %聚类分析
         if (nCol == size(data,2))
             D = data;  %如果没有全0列样本，那么D就取原来的数值
         else
             [~,C]  = litekmeans(data',nCol); %matlab有这个函数的，作用是‘剔除全0列’后按行重新进行聚类，聚nCol个类，
                                              %依据题意是把每类选出的6个样本聚类成D矩阵中4个原子，不稳定性在于选中的样本如果有超过两个全0列存在，那么就会出错
                                              %经过猜想并且代码验证，这条命令是把相同的类，聚类之后对应特征的值求平均得到新的类该特征的特征值
              D       = C';
         end    
    case {'sequence'}  %重新打散并随机列排序，剔除全0行之后的可能D矩阵的列会减少
         perm     = randperm(size(data,2)); 
         D        = data(:, perm(1:nCol));
    otherwise 
        error{'Unkonw method.'}
end
D        = normcols(D); %列平方归一，就是normalize，main程序没有用集成命令，却用公式，怀疑到底是不是同一个人写的代码，再吐槽一遍
end      