run(fullfile('vl_setupnn.m')) ;
rootPath = pwd;
trainSet = 'LIB';%'YOS','ND','HP'
flagCS = 1;
flagAug = 1;
flagGPU = 1;
batchSize = 1000;

if flagCS   %if CS structure is used, testdata should be in size of 64*64*1*N 
    testPatch = randi([0 255],64,64,1,10,'single');%the pixel value of the input patches shoud be in range of 0 tp 255
else
    testPatch = randi([0 255],32,32,1,10,'single');
end

desFloat = cal_L2Net_des(rootPath,trainSet,flagCS,flagAug,testPatch,batchSize,flagGPU);%output a 128(or 256)*N matrix, each colum is a descriptor

desBinary = desFloat;
desBinary(find(desBinary>0)) = 1;
desBinary(find(desBinary<=0)) = -1;
