# Multi-label
implementation of the paper "ML-MG Multi-label Learning with Missing Labels Using a Mixed Graph"

实验结果：
    因为训练集中缺失标签是随机的，我只对测试集进行了结果评判。在不加语义约束的情况下，AP率最高为0.16以上，加语义约束的情况下，AP率最高为0.3以上

遇到的问题：
（1）AP率比较低，在不同缺失比例下，不加语义约束时，论文中的AP率为0.3到0.4左右，而我的代码在不加语义约束的情况下仅为0.16左右
（2）论文中的AP率最高也仅为0.5左右，而当我把所有标签都视为负标签的时候，AP率就能达到0.5 ( 即相当于随机猜测 )，所以我对本论文工作的意义有些疑惑
