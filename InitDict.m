function [Dict, DLabel] = InitDict(trDat, trLab, C, options)  %trdat�ǳ�����15��*6��/��ľ���trLab�Ƕ�Ӧ�ڵ�C���ȡ��tr_dat�е�������C�����������options��һ������Ľṹ��
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
        pDSizeList = repmat(pDSizeList, [1,C]);%���ֻ��һ�У��Ͱ���������Ƴ�1*C��15���ľ���
    end
    if (size(pDSizeList,2) < C)
         error('The pDSizeList is not specified');
    end
else
    error('The pDSizeList is not specified');
end     %��ΰ�pDSizeList����������Ƴ���[1,C]�ľ���

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
end            %��Щ�ܵ���˵���жϽṹ�����䣬���˸��ƾ��������ֵ�����⣬�����ֻ��Ϊ�˳���Ľ�׳��д

indDictSize = sum(pDSizeList); %������15��*ÿһ��4��ԭ��
Dict        = zeros(size(trDat,1),indDictSize+shDSize); %��ʼ��576*��15*4+5)��ȫ0����indDictSize��15*4��shDSize�ǹ���ԭ�Ӹ���
DLabel      = zeros(1, indDictSize + shDSize); %---------------ֻ��˵1*65��ȫ0��Ŀǰû�и����ε���⣡��������ʱ�İ�������-------------------

%% initilize the dictionary
%initilize the whole dictionary with randomly selected samples
if(strcmpi(wayInit,'wholeRand')) %ȫ�����������ԭ�Ӻ͹���ԭ��ȫ�������kmwans���޼ල�ľ����㷨�����ĵ���������������ϣ��ܼ�
    nData = size(trDat,2); %ѡȡ������90��
    nAtom = size(Dict,2); %D����65��
    Dict  = trDat(:, randperm(nData, nAtom)); %����������������ȡ65�в�������ȫ���,ע�⣬�Ѿ���һ��Ŷ
    pos   = 1;
    for c = 1:C
         perDSize = pDSizeList(c); %���İɣ���Ȼ��repmat�ģ�ÿһ����һ����
         DLabel(pos: pos+perDSize-1) = c; %һ��60�У�4��1,4��2��һֱ��4��15
         pos = pos + perDSize; %pos=pos+D�������ÿһ�����ԭ�ӵ�������repmat֮��һ����
    end
    DLabel(pos: pos+shDSize-1) = C+1; %����ԭ�ӷ������
    Dict = normcols(Dict); %����ƽ���͹�һ���ؼ�ǰ��û��������������ù�ʽ��һ�ġ��������ˣ���һ����д����
    return;
end

%initialize each part-dictionaries ��γ�ʼ�����Ƕ�Ӧÿһ���ԭ�ӣ�����һ���г�ʼ������ԭ��
pos = 1;
for c = 1:C
    partDsize = pDSizeList(c);  %ÿһ����pDSizeList��ԭ�ӣ����İɣ���Ȼ��repmat�ģ�ÿһ�඼һ����
    fprintf('\tClass-%d''s part-dictionary initialization...\n',c); %ÿ�ҳ�һ�����ʾһЩ�ַ�
    col_ids  = find(trLab == c);  %�����з���������1-15��λ�ã�����90�������ķ������ݣ���Quora�Ͽ�������matrix����ѭ���ĵ������ӣ�
                                  %matlabִ��for���ġ��������ԭ��֮һ�ǰ��д���Ԫ�ص���ָ����һ����ӿռ临�Ӷ�
    % ensure no zero data elements are chosen
    data_ids = find(sum(trDat(:,col_ids).^2,1) > 1e-6);  %��Ӧ�Ĵ�1��15�࣬����������޳������������� ensure no zero data elements are chosen
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
        Dpart = Dictionary_Ini(trDat(:,col_ids(data_ids)),partDsize,wayInit);  %��һ��������ʾ���������ѡ����������������������
                                                                               %�ڶ�����ÿһ�������ĸ�������������ģʽѡ��ľ��
    end
    Dict(:, pos:pos+partDsize-1)  = Dpart;  %�Զ�Ӧ��C����ֵ�����ԭ�ӽ��и�ֵ
    DLabel(pos : pos+partDsize-1) = c;  %��Ӧ��ԭ�ӣ�ԭ�Ӿ����У����б�ע�ڼ���
    pos = pos+partDsize;  %������⣬����������������
end %����Ϊֹǰ��60�еķ���ԭ��

% initialize the shared atoms  ��ʼ��������ԭ��
if( shDSize > 0)  %shDSize���ǹ���ԭ�ӵĸ���
    fprintf('\tThe shared part initialization...\n');    
    
    % substract the individual component
    resData=trDat;  %����ѵ�������ľ��󣬾���ÿ�������ѡ6�����������Ǹ�
    for c = 1:C
        indData   = resData(:,trLab == c);  %ѡ����c���ԭ��
        partD     = Dict(:,DLabel == c);  %�ȸ���D����ĵ�c��ԭ�ӵ�
        partX     = (partD'*partD + options.beta * eye(size(partD,2)))\(partD'*indData); %shared atom��ʽ
        resData(:,trLab == c) = indData - partD * partX; %ͬ�ϵĹ�ʽ
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
            shD = Dictionary_Ini(resData(:,data_ids), shDSize, wayInit); %���������޳�ȫ0�к�ֵ��ֻҪû��ȫ0ԭ�ӣ��Ͳ���
       end
    else
        shD = trDat(:,1:shDsize); %�����������ȫ0��ֻ��һ�л������£��͸�ֵǰ5�У�����
    end
    Dict(:, end+1-shDSize:end) = shD; 
    DLabel(end+1-shDSize:end)  = C+1;
end
end

function D    =    Dictionary_Ini(data, nCol, wayInit) %��һ����������ѡ���ķ����������󣬵ڶ�����ÿһ�������ĸ�������������ģʽѡ��ľ��
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
m   =    size(data,1); %���������������ÿ��ԭ���ж��ٸ���ȡ������
switch lower(wayInit)
    case {'pca'}  %��ģ���������������ɷַ���
        [D,disc_value,Mean_Image]   =    Eigenface_f(data,nCol-1);
         D                          =    [D Mean_Image];
    case {'random'} %���
         D                          =    randn(m, nCol);
    case {'kmeans'} %�������
         if (nCol == size(data,2))
             D = data;  %���û��ȫ0����������ôD��ȡԭ������ֵ
         else
             [~,C]  = litekmeans(data',nCol); %matlab����������ģ������ǡ��޳�ȫ0�С��������½��о��࣬��nCol���࣬
                                              %���������ǰ�ÿ��ѡ����6�����������D������4��ԭ�ӣ����ȶ�������ѡ�е���������г�������ȫ0�д��ڣ���ô�ͻ����
                                              %�������벢�Ҵ�����֤�����������ǰ���ͬ���࣬����֮���Ӧ������ֵ��ƽ���õ��µ��������������ֵ
              D       = C';
         end    
    case {'sequence'}  %���´�ɢ������������޳�ȫ0��֮��Ŀ���D������л����
         perm     = randperm(size(data,2)); 
         D        = data(:, perm(1:nCol));
    otherwise 
        error{'Unkonw method.'}
end
D        = normcols(D); %��ƽ����һ������normalize��main����û���ü������ȴ�ù�ʽ�����ɵ����ǲ���ͬһ����д�Ĵ��룬���²�һ��
end      