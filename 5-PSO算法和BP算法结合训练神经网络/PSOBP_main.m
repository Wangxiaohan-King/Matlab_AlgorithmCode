%PSO算法优化BP神经网络
%介绍详见《MATLAB在数学建模中的应用》P88
%本案例网络结构为三层，基于PSO算法和BP算法先后训练神经网络的权值和阀值（不优化网络的结构），然后逼近一个函数
%这是主函数，命名为PSOBP502.m
%这个程序很长，运行完毕大约花费10分钟的时间
% function main
clc;
clear all;
close all;

MaxRunningTime=1; %该参数是为了使网络集成，重复训练MaxRunningTime次
HiddenUnitNum=12;
rand('state',sum(100*clock)); %rand产生的是伪随机数，状态相同产生的随机数就一样，这里设置状态不同
TrainSamIn=-4:0.07:2.5;
TrainSamOut=1.1*(1-TrainSamIn+2*TrainSamIn.^2).*exp(-TrainSamIn.^2/2);
TestSamIn=2:0.04:3;
TestSamOut=1.1*(1-TestSamIn+2*TestSamIn.^2).*exp(-TestSamIn.^2/2);
[xxx,TrainSamNum]=size(TrainSamIn);
[xxx,TestSamNum]=size(TestSamIn);
% for HiddenUnitNum=3:MaxHiddenLayerNode %隐含层神经元的个数可以取逐渐增大的合理整数
    fprintf('\n the hidden layer node');HiddenUnitNum
    TrainNNOut=[];
    TestNNOut=[];
    for t=1:MaxRunningTime
        fprintf('the current running times is');t
        [NewW1,NewB1,NewW2,NewB2]=PSOTrain(TrainSamIn,TrainSamOut,HiddenUnitNum);
        disp('PSO算法训练神经网络结束，BP算法接着训练网络……');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BP算法参数初始化，注意与上面PSO参数一致
SamInNum=length(TrainSamIn);
TestSamNum=length(TestSamIn);
InDim=1;
OutDim=1;
%学习样本添加噪声
rand('state',sum(100*clock))
NoiseVar=0.01;
Noise=NoiseVar*randn(1,SamInNum);
SamIn=TrainSamIn;
SamOutNoNoise=TrainSamOut;
SamOut=SamOutNoNoise + Noise;
MaxEpochs=300;      %BP算法训练次数
lr=0.003;           %学习速率
E0=0.0001;          %最小允许误差
W1=NewW1;
B1=NewB1;
W2=NewW2';
B2=NewB2;
%%%%%%%%直接用BP算法测试时，神经网络的初始参数，此时MaxEpochs=20000训练效果挺好
% W1=0.2*rand(HiddenUnitNum,InDim)-0.1;          %输入层到隐层的初始权值
% B1=0.2*rand(HiddenUnitNum,1)-0.1;              %隐节点初始阈值
% W2=0.2*rand(OutDim,HiddenUnitNum)-0.1;         %隐层到输出层的初始权值
% B2=0.2*rand(OutDim,1)-0.1;                     %输出层初始阈值
%%%%%%%%
W1Ex=[W1 B1]; %12*2
W2Ex=[W2 B2]; %1*13
SamInEx=[SamIn' ones(SamInNum,1)]'; %2*93
ErrHistory=[];
%网络参数初始化完毕

for i=1:MaxEpochs
    HiddenOut=logsig(W1Ex*SamInEx);    %12*93；前面几个矩阵的处理，都是为了这一步的矩阵运算
    HiddenOutEx=[HiddenOut' ones(SamInNum,1)]';  %13*93
    NetworkOut=W2Ex*HiddenOutEx;  %1*93
    Error=SamOut-NetworkOut;  %1*93
    %这仅限于输出是一维的情况
    SSE=sumsqr(Error);  %平方和
    %记录每次权值调整后的训练误差
    ErrHistory=[ErrHistory SSE];
 
    if SSE<E0,break, end    %代码对应公式见：周志华《机器学习》P103
        %计算反向传播误差
        Delta2=Error;    %相当于gi
        Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut); %相当于eh
        %计算权值调整量
        dW2Ex=Delta2*HiddenOutEx'; %求出了隐层-输出权值△w_hj和输出神经元阈值△θ_j
        dW1Ex=Delta1*SamInEx';     %求出了输入层-隐层权值△v_ih和隐层神经元阈值△r_j
        %权值调整
        W1Ex=W1Ex+lr*dW1Ex;
        W2Ex=W2Ex+lr*dW2Ex;
        %分离隐层到输出层的权值
        W2=W2Ex(:,1:HiddenUnitNum);
    
end
    
W2=W2Ex(:,1:HiddenUnitNum);
W1=W1Ex(:,1:InDim);
B1=W1Ex(:,InDim+1);
B2=W2Ex(:,1+HiddenUnitNum); 

TrainHiddenOut=logsig(W1*SamIn+repmat(B1,1,SamInNum));
TrainNNOut=W2*TrainHiddenOut+repmat(B2,1,SamInNum);
TestHiddenOut=logsig(W1*TestSamIn+repmat(B1,1,TestSamNum));
TestNNOut=W2*TestHiddenOut+repmat(B2,1,TestSamNum);

figure(MaxEpochs+1);
hold on;
grid;
h1=plot(SamIn,SamOut); %训练样本带噪声输出
set(h1,'color','r','linestyle','-',...
    'linewidth',2.5,'marker','p','markersize',5);
hold on 
h2=plot(TestSamIn,TestSamOut); %测试样本真实输出
set(h2,'color','g','linestyle','--',...
    'linewidth',2.5,'marker','^','markersize',7);
h3=plot(SamIn,TrainNNOut);  %训练样本神经网络拟合输出
set(h3,'color','c','linestyle','-.',...
    'linewidth',2.5,'marker','o','markersize',5);
h4=plot(TestSamIn,TestNNOut); %测试样本神经网络拟合输出
set(h4,'color','m','linestyle',':',...
    'linewidth',2.5,'marker','s','markersize',5);
xlabel('Input x','fontsize',13);ylabel('Output y','fontsize',13);
box on;axis tight;
%title('PSO-BP神经网络误差测试图');
legend('网络学习实际样本值','网络测试实际样本值',...
    '网络学习网络输出值','网络测试网络输出值');
hold off;
    end
% end
fidW1=fopen('W1.txt','a+');fidB1=fopen('B1.txt','a+');
fidW2=fopen('W2.txt','a+');fidB2=fopen('B2.txt','a+');
for i=1:length(W1)
    fprintf(fidW1,'\n %6.5f',W1(i));
end
for i=1:length(B1)
    fprintf(fidB1,'\n %6.5f',B1(i));
end
for i=1:length(W2)
    fprintf(fidW2,'\n %6.5f',W2(i));
end
for i=1:length(B2)
    fprintf(fidB2,'\n %6.5f',B2(i));
end
fclose(fidW1);fclose(fidB1);fclose(fidW2);fclose(fidB2);
