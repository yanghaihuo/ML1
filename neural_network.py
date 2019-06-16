'''
反向传播BP算法的特征是利用输出后的误差来估计输出层的直接前导层的误差，再用这个误差
估计更前一层的误差，如此一层一层的反向传播下去，就获得了所有其它各层误差估计。BP算
法的学习过程由信号的正向传播与误差的逆向传播两个过程组成
'''

# 使用神经网络算法预测销量高低
import pandas as pd

#参数初始化
inputfile = '../data/sales_data.xls'
data = pd.read_excel(inputfile, index_col = u'序号') #导入数据

#数据是类别标签，要将它转换为数据
#用1来表示“好”、“是”、“高”这三个属性，用0来表示“坏”、“否”、“低”
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = 0
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim = 3, output_dim = 10))
model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(input_dim = 10, output_dim = 1))
model.add(Activation('sigmoid')) #由于是0-1输出，用sigmoid函数作为激活函数

#model.compile(loss = 'binary_crossentropy', optimizer = 'adam', class_mode = 'binary')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy以及模式为binary
#另外常见的损失函数还有mean_squared_error、categorical_crossentropy
# 求解方法我们指定用adam还有sgd、rmsprop

#训练模型，学习一千次
model.fit(x, y, nb_epoch = 1000, batch_size = 10)
#分类预测
yp = model.predict_classes(x).reshape(len(y))


#导入自行编写的混淆矩阵可视化函数
def cm_plot(y, yp):
  #导入混淆矩阵函数
  from sklearn.metrics import confusion_matrix
  #混淆矩阵
  cm = confusion_matrix(y, yp)
  #导入作图库
  import matplotlib.pyplot as plt
  #画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
  plt.matshow(cm, cmap=plt.cm.Greens)
  #颜色标签
  plt.colorbar()
  #数据标签
  for x in range(len(cm)):
    for y in range(len(cm)):
      plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center',
                   verticalalignment='center')
  #坐标轴标签
  plt.ylabel('True label')
  #坐标轴标签
  plt.xlabel('Predicted label')
  return plt

#显示混淆矩阵可视化结果
cm_plot(y,yp).show()
