import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC

# 导入词向量为训练特征
x = np.load('TrainingSet/x_train_vec.npy')

# 导入情绪分类作为目标特征
y = np.load('TrainingSet/y_train.npy')

# 构建支持向量机分类模型
model = SVC(kernel='rbf', verbose=True)

# 训练模型
model.fit(x, y)

# 保存模型为二进制文件
joblib.dump(model, 'SVM/svm_model.pkl')