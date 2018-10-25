load glass_dataset
glassdata=glassInputs;

%gass_dataset���Է�Ϊ2��;
%��1��163������Ϊ��һ�࣬��164-214��Ϊ�ڶ���;
%��glass_dataset���Ϊ��������traindata��testdata;
%����һ�����ݵĺ�50�����ڲ��ԣ��ڶ������ݵĺ�20�����ڲ��ԣ��鵽testdata,��������ѵ�����鵽traindata;
traindata1=glassdata(1:9,1:113);
testdata1=glassdata(1:9,114:163);
traindata2=glassdata(1:9,164:194);
testdata2=glassdata(1:9,195:214);
traindata=[traindata1 traindata2];
testdata=[testdata1 testdata2];

%����glass_datasetÿһ�ࡢÿһά��������̬�ֲ�;
%�ֱ����2����9�ֳɷֵľ�ֵ�ͷ���;
for i=1:9
    [mu(i,1),sigma(i,1)]=normfit(traindata1(i,:));
    [mu(i,2),sigma(i,2)]=normfit(traindata2(i,:));
end

%�����ر�Ҷ˹����������Ȼ����;
%����evidenceһ�£��������=��Ȼ����*�������;
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

%���������ȷ�ͷ������ĸ��ʣ�
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
%��ͼ���ֱ�չʾ9ά���ݵķ����������ɫ��ͬΪͬһ�ࣩ;
%��һ������Ϊ��ɫ���ڶ�������Ϊ��ɫ;
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

