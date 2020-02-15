import pandas as pd
import jieba
import numpy as np

# 加载语料库文件，并导入数据
neg = pd.read_excel('TrainingSet/neg.xls', header=None, index=None)
pos = pd.read_excel('TrainingSet/pos.xls', header=None, index=None)

# jieba 分词
word_cut = lambda x: jieba.lcut(x)
pos['words'] = pos[0].apply(word_cut)
neg['words'] = neg[0].apply(word_cut)

# 使用 1 表示积极情绪，0 表示消极情绪，并完成数组拼接
x = np.concatenate((pos['words'], neg['words']))
y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

# 将 Ndarray 保存为二进制文件备用
np.save('TrainingSet/x_train.npy', x)
np.save('TrainingSet/y_train.npy', y)

print('done.')