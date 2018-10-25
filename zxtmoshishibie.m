load glass_dataset
glassdata=glassInputs;

%gass_dataset可以分为2类;
%第1—163列数据为第一类，第164-214列为第二类;
%将glass_dataset拆分为两个矩阵：traindata和testdata;
%将第一类数据的后50列用于测试，第二类数据的后20列用于测试，归到testdata,其余用于训练，归到traindata;
traindata1=glassdata(1:9,1:113);
testdata1=glassdata(1:9,114:163);
traindata2=glassdata(1:9,164:194);
testdata2=glassdata(1:9,195:214);
traindata=[traindata1 traindata2];
testdata=[testdata1 testdata2];

%假设glass_dataset每一类、每一维都服从正态分布;
%分别计算2类中9种成分的均值和方差;
for i=1:9
    [mu(i,1),sigma(i,1)]=normfit(traindata1(i,:));
    [mu(i,2),sigma(i,2)]=normfit(traindata2(i,:));
end

%用朴素贝叶斯方法处理似然函数;
%由于evidence一致，后验概率=似然函数*先验概率;
posterior=zeros(2,70);
for i=1:70 
    for j=1:2
        if j==1
           priori(j)=163/214;
        elseif j==2
            priori(j)=51/214;
        end
        likelihood(j)=1;
        for d=1:9
            likelihood(j)=likelihood(j)*normpdf(testdata(d,i),mu(d,j),sigma(d,j));
        end
        posterior(j,i)=likelihood(j)*priori(j);
    end
    [c,s]=max(posterior(:,i)');
    category(i)=s;
end

%计算分类正确和分类错误的概率；
category_true1=0;
category_false1=0;
category_true2=0;
category_false2=0;
for k=1:50
    if category(k)==1
        category_true1=category_true1+1;
    else
        category_false1=category_false1+1;
    end
end
for k=51:70
    if category(k)==2
        category_true2=category_true2+1;
    else
        category_false2=category_false2+1;
    end
end
category_false=category_false1+category_false2;
category_true=category_true1+category_true2;
false_probability=category_false/70;
true_probability=category_true/70;
%作图，分别展示9维数据的分类情况（颜色相同为同一类）;
%第一类数据为蓝色，第二类数据为绿色;
figure ('name','model1')
hold on
scatter(1:70,testdata(1,:),30,category,'filled');
plot([50,50],[0,20],'--k');
xlabel('sample');
ylabel('A');
hold off
figure ('name','model2')
hold on
scatter(1:70,testdata(2,:),30,category,'filled');
plot([50,50],[0,20],'--k');
xlabel('sample');
ylabel('B');
figure ('name','model3')
hold on
scatter(1:70,testdata(3,:),30,category,'filled');
plot([50,50],[0,5],'--k');
xlabel('sample');
ylabel('C');
hold off
figure ('name','model4')
hold on
scatter(1:70,testdata(4,:),30,category,'filled');
plot([50,50],[0,40],'--k');
xlabel('sample');
ylabel('D');
hold off
figure ('name','model5')
hold on
scatter(1:70,testdata(5,:),30,category,'filled');
plot([50,50],[0,150],'--k');
xlabel('sample');
ylabel('E');
hold off
figure ('name','model6')
hold on
scatter(1:70,testdata(6,:),30,category,'filled');
plot([50,50],[0,5],'--k');
xlabel('sample');
ylabel('F');
hold off
figure ('name','model7')
hold on
scatter(1:70,testdata(7,:),30,category,'filled');
plot([50,50],[0,20],'--k');
xlabel('sample');
ylabel('G');
hold off
figure ('name','model8')
hold on
scatter(1:70,testdata(8,:),30,category,'filled');
plot([50,50],[0,5],'--k');
xlabel('sample');
ylabel('H');
hold off
figure ('name','model9')
hold on
scatter(1:70,testdata(9,:),30,category,'filled');
plot([50,50],[0,10],'--k');
xlabel('sample');
ylabel('I');
hold off

