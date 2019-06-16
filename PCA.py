import pandas as pd


inputfile = '../data/principal_component.xls'

data = pd.read_excel(inputfile, header = None) #读入数据

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data)
print(pca.components_ )#返回模型的各个特征向量  输出主成分，即行数为降维后的维数，列数为原始特征向量转换为新特征的系数
print(pca.explained_variance_) # 新特征 每维所能解释的方差大小
print(pca.explained_variance_ratio_ )#返回各个成分各自的方差百分比 新特征 每维所能解释的方差大小在全方差中所占比例



# pca = PCA(n_components=3)  # 加载PCA算法，设置降维后主成分数目为3
# reduced_data = pca.fit_transform(data)  # 对样本进行降维
# print(reduced_data)
