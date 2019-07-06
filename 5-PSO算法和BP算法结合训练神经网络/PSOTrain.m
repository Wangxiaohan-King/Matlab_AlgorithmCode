function [NewW1,NewB1,NewW2,NewB2]=PSOTrain(SamIn,SamOut,HiddenUnitNum)
% %%%调试%%%
% SamIn=-4:0.07:2.5;
% SamOut=1.1*(1-SamIn+2*SamIn.^2).*exp(-SamIn.^2/2);
% HiddenUnitNum=12;
% %%%
Maxgeneration=700;  %最大迭代次数
E0=0.0001;          %最小允许误差
Xmin=-10;           %粒子位置、速度的范围
Xmax=10;
Vmin=-5;
Vmax=5;
M=100;              %粒子规模
c1=2.7;             %加速度系数
c2=1.3;
w=0.9;              %惯性系数
[R,SamNum]=size(SamIn);  %R是代表输入个数对应输入神经元个数，一行输入即R=1代表输入层只有一个神经元
[S2,SamNum]=size(SamOut);%R是代表输出个数对应输出神经元个数，一行输出及S2=1代表输出层只有一个神经元
generation=1;
Done=0;
                                   %初始化M个(HiddenUnitNum)*(R+S2+1)的全0矩阵；
Pb1=zeros(HiddenUnitNum,R+S2+1,M); %Pb1我猜这是（输入与隐层神经元直接连接权值W_ih、隐层与输出神经元直接连接权值W_ho、隐层神经元的阈值b_h）的局部最优值
Pb2=zeros(S2,M);                   %Pb2我猜这是（输出层神经元阈值）的局部最优值
Pg1=zeros(HiddenUnitNum,R+S2+1);   %Pg1我猜这是（W_ih、W_ho、b_h）的全局最优值
Pg2=zeros(S2,1);                   %Pg2我猜这是（输出层神经元阈值）的全局最优值
E=zeros(size(SamOut)); %我猜存放误差
rand('state',sum(100*clock));
startP1=rand(HiddenUnitNum,R+S2+1,M)-0.5;%W_ih、W_ho、b_h对应粒子的初始位置
startP2=rand(S2,M)-0.5;                  %输出层神经元阈值对应粒子的初始位置
startV1=rand(HiddenUnitNum,R+S2+1,M)-0.5;%W_ih、W_ho、b_h对应粒子的初始速度
startV2=rand(S2,M)-0.5;                  %输出层神经元阈值对应粒子的初始速度
endP1=zeros(HiddenUnitNum,R+S2+1,M);     %W_ih、W_ho、b_h对应粒子的更新后的位置
endP2=zeros(S2,M);                       %输出层神经元阈值对应粒子的更新后的位置
endV1=zeros(HiddenUnitNum,R+S2+1,M);     %W_ih、W_ho、b_h对应粒子的更新后速度
endV2=zeros(S2,M);                       %输出层神经元阈值对应粒子的更新后速度
startE=zeros(1,M);
endE=zeros(1,M);
for i=1:M  %每个粒子包含神经网络完整的参数W1、W2、B1、B2；一共有M个粒子
    W1=startP1(1:HiddenUnitNum,1:R,i);   %给每个粒子初始化不同的神经网络参数
    W2=startP1(1:HiddenUnitNum,R+1:R+S2,i);
    B1=startP1(1:HiddenUnitNum,R+S2+1,i);
    B2=startP2(1:S2,i);
    for q=1:SamNum
        TempOut=logsig(W1*SamIn(:,q)+B1);
        NetworkOut(1,q)=W2'*TempOut+B2;
    end
    E=NetworkOut-SamOut;
    startE(1,i)=sumsqr(E)/(SamNum*S2); %sumsqr函数：求平方和
    Pb1(:,:,i)=startP1(:,:,i);  %每个粒子的初始位置当做初始的局部最优
    Pb2(:,i)=startP2(:,i);
end
[val,position]=min(startE(1,:));
Pg1=startP1(:,:,position);  %初始化全局最优
Pg2=startP2(:,position);
Pgvalue=val;
Pgvalue_last=Pgvalue;

while(~Done)
    for num=1:M
        endV1(:,:,num)=w*startV1(:,:,num)+c1*rand*(Pb1(:,:,num)-startP1(:,:,num))+c2*rand*(Pg1-startP1(:,:,num));
        endV2(:,num)=w*startV2(:,num)+c1*rand*(Pb2(:,num)-startP2(:,num))+c2*rand*(Pg2-startP2(:,num));
        for i=1:HiddenUnitNum
            for j=1:(R+S2+1)
                endV1(i,j,num)=endV1(i,j,num);
                if endV1(i,j,num)>Vmax
                    endV1(i,j,num)=Vmax;
                elseif endV1(i,j,num)<Vmin
                        endV1(i,j,num)=Vmin;
                end
            end
        end
        for s2=1:S2
            endV2(s2,num)=endV2(s2,num);
            if endV2(s2,num)>Vmax
               endV2(s2,num)=Vmax;
            elseif endV2(s2,num)<Vmin
                   endV2(s2,num)=Vmin;
            end
        end
        endP1(:,:,num)=startP1(:,:,num)+endV1(:,:,num);
        endP2(:,num)=startP2(:,num)+endV2(:,num);
        for i=1:HiddenUnitNum
            for j=1:(R+S2+1)
                if endP1(i,j,num)>Xmax
                   endP1(i,j,num)=Xmax;
                elseif endP1(i,j,num)<Xmin
                       endP1(i,j,num)=Xmin;
                end
            end
        end
        for s2=1:S2
            if endP2(s2,num)>Xmax
               endP2(s2,num)=Xmax;
            elseif endP2(s2,num)<Xmin
               endP2(s2,num)=Xmin;
            end
        end  
        W1=endP1(1:HiddenUnitNum,1:R,num);
        W2=endP1(1:HiddenUnitNum,R+1:R+S2,num);
        B1=endP1(1:HiddenUnitNum,R+S2+1,num);
        B2=endP2(1:S2,num);
        for q=1:SamNum
            TempOut=logsig(W1*SamIn(:,q)+B1);
            NetworkOut(1,q)=W2'*TempOut+B2;
        end
        E=NetworkOut-SamOut;
        SSE=sumsqr(E);   %便于在命令窗口观察网络误差的变化情况
        endE(1,num)=sumsqr(E)/(SamNum*S2);
        if endE(1,num)<startE(1,num)
            Pb1(:,:,num)=endP1(:,:,num);
            Pb2(:,num)=endP2(:,num);
            startE(1,num)=endE(1,num);
        end
    end
    w=0.9-(0.5/Maxgeneration)*generation;
    [value,position]=min(startE(1,:));
    if value<Pgvalue
        Pg1=Pb1(:,:,position);
        Pg2=Pb2(:,position);
        Pgvalue=value;
    end
    if (generation>=Maxgeneration)
        Done=1;
    end
    if Pgvalue<E0
        Done=1;
    end
    
    startP1=endP1;
    startP2=endP2;
    startV1=endV1;
    startV2=endV2;
    startE=endE;
    generation=generation+1;
end
W1=Pg1(1:HiddenUnitNum,1:R);
W2=Pg1(1:HiddenUnitNum,R+1:R+S2);
B1=Pg1(1:HiddenUnitNum,R+S2+1);
B2=Pg2(:,1);
NewW1=W1;
NewW2=W2;
NewB1=B1;
NewB2=B2;
