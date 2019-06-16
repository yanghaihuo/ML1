'''
特征筛选
    线性关系：
         F检验：选择F值大的或者p值小的特征
         递归特征消除(RFE RFECV交叉验证)
         稳定性选择：随机lasso和随机逻辑回归
    非线性关系：
         决策树
         神经网络
'''

import pandas as pd

filename = '../data/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR # 随机逻辑回归特征筛选
rlr = RLR() # 建立随机逻辑回归模型筛选变量 默认阈值0.25 也可以手动设置 RLR(selection_threshold=0.5)
rlr.fit(x, y)
rlr.get_support() # 获取特征筛选结果
print(u'有效特征为：%s' % ','.join(data.columns[rlr.get_support()]))
x = data[data.columns[rlr.get_support()]].as_matrix() # 筛选好特征

lr = LR()
lr.fit(x, y) # 用筛选后的特征数据来训练模型
print(u' 模型的平均正确率为：%s' % lr.score(x, y))