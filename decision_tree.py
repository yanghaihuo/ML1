'''
ID3算法具体流程：
1 对当前样本集合，计算所有属性的信息增益
2 选择信息增益最大的属性作为测试属性，把测试属性取值相同的样本划为同一个子样本集
3 若子样本集的类别属性只含有单个属性，则分支为叶子节点，判断其属性值并标上相应的
  符号，然后返回调用处；否则对子样本集递归调用本算法

'''

# 使用ID3决策树算法预测销量高低
import pandas as pd

inputfile = '../data/sales_data.xls'
data = pd.read_excel(inputfile, index_col = u'序号')

data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy')
dtc.fit(x, y)

from sklearn.tree import export_graphviz
x = pd.DataFrame(x)
from sklearn.externals.six import StringIO
x = pd.DataFrame(x)
with open("tree.dot", 'w') as f:
  f = export_graphviz(dtc, feature_names = x.columns, out_file = f)
