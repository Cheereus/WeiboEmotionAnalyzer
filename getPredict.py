from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
from sklearn.externals import joblib
import pandas as pd

keyword = '奔驰男'

# 读取 Word2Vec 并对新输入进行词向量计算
def average_vec(words):
    # 读取 Word2Vec 模型
    w2v = Word2Vec.load('TrainingSet/w2v_model.pkl')
    vec = np.zeros(300).reshape((1, 300))
    for word in words:
        try:
            vec += w2v[word].reshape((1, 300))
        except KeyError:
            continue
    return vec

# 对微博正文进行情感判断
def svm_predict():
    # 读取微博数据
    df = pd.read_csv("DataSet/ipad3.csv", header=0)

    comment_sentiment = []
    for string in df['sentence']:
        # 对内容分词
        words = jieba.lcut(str(string))
        words_vec = average_vec(words)
        # 读取支持向量机模型
        model = joblib.load('SVM/svm_model.pkl')
        result = model.predict(words_vec)
        comment_sentiment.append(result[0])

        # 实时返回积极或消极结果
        if int(result[0]) == 1:
            print(string, '[积极]')
        else:
            print(string, '[消极]')

    #将情绪结果合并到原数据文件中
    merged = pd.concat([df, pd.Series(comment_sentiment, name='用户情绪')], axis=1)

    # 储存文件
    pd.DataFrame.to_csv(merged, 'Result/prediction_ipad3.csv')
    print('done.')

# 执行
svm_predict()