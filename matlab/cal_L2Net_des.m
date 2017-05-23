function out = cal_L2Net_des(rootPath,trainSet,flagCS,flagAug,testPatch,batchSize,flagGPU)
   if strcmp(trainSet,'YOS')
        if flagAug
            netName = fullfile(rootPath,'L2Net-YOS+.mat')
        else
            netName = fullfile(rootPath,'L2Net-YOS.mat')
        end
    end
    if strcmp(trainSet,'ND')
        if flagAug
            netName = fullfile(rootPath,'L2Net-ND+.mat')
        else
            netName = fullfile(rootPath,'L2Net-ND.mat')
        end
    end
    if strcmp(trainSet,'LIB')
        if flagAug
            netName = fullfile(rootPath,'L2Net-LIB+.mat')
        else
            netName = fullfile(rootPath,'L2Net-LIB.mat')
        end
    end
    if strcmp(trainSet,'HP')
        if flagAug
            netName = fullfile(rootPath,'L2Net-HP+.mat')
        else
            netName = fullfile(rootPath,'L2Net-HP.mat')
        end
    end
load(netName)

for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type, 'bnormPair')
        net.layers{i}.type = 'bnorm';
    end
end
for i=1:numel(netCen.layers)
    if strcmp(netCen.layers{i}.type, 'bnormPair')
        netCen.layers{i}.type = 'bnorm';
    end
end

if flagGPU
    net = vl_simplenn_move(net, 'gpu') ;
    netCen = vl_simplenn_move(netCen, 'gpu') ;
end


numPatch = size(testPatch,4);
if flagCS
    if size(testPatch,1) == 64
        testPatchCen = testPatch(17:48,17:48,1,:);
        testPatch = imresize(testPatch,0.5);
    else
        error('patch size does match')
        return
    end
    testPatchCen = bsxfun(@minus, testPatchCen, pixMeanCen);
    z = reshape(testPatchCen,[],numPatch) ;
    z = bsxfun(@minus, z, mean(z,1)) ;
    n = std(z,0,1) ;
    z = bsxfun(@times, z, 1 ./ (n+1e-12)) ;
    testPatchCen = reshape(z, 32, 32, 1, []) ;
end
testPatch = bsxfun(@minus, testPatch, pixMean);
z = reshape(testPatch,[],numPatch) ;
z = bsxfun(@minus, z, mean(z,1)) ;
n = std(z,0,1) ;
z = bsxfun(@times, z, 1 ./ (n+1e-12)) ;
testPatch = reshape(z, 32, 32, 1, []) ;

idxMax = ceil(numPatch/batchSize);
out = [];
for idx = 1:idxMax
    sta = (idx-1)*batchSize+1;
    en = min([idx*batchSize,numPatch]);
    dataTemp = testPatch(:,:,:,sta:en);
    if flagGPU
        dataTemp = gpuArray(dataTemp);
    end
    res=vl_simplenn(net,dataTemp,[],[],'mode','test','conserveMemory',1,'cudnn',1);
    if flagCS
        dataTemp = testPatchCen(:,:,:,sta:en);
        if flagGPU        
            dataTemp = gpuArray(dataTemp);
        end
        resCen=vl_simplenn(netCen,dataTemp,[],[],'mode','test','conserveMemory',1, 'cudnn',1);
        out = [out, gather(squeeze(cat(3,res(end).x,resCen(end).x)))];
    else
        out = [out, gather(squeeze(res(end).x))];
    end
end

