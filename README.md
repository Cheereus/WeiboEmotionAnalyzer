# WeiboCommentsEmotionAnalyzer

微博评论爬取并进行情感分析，原创的部分不多，主要是将他人成果进行了串联，形成了完整流程，修复了一些小问题

环境为 Anaconda (Python 3.7.1 64-bit)

## 相关

jieba

pandas

word2vec

svm

sklearn

gensim

## 参考内容

1. 爬虫部分主要参考 <https://github.com/Python3Spiders/WeiboSuperSpider>

2. word2Vec 和支持向量机训练的部分主要参考 <https://m-zhoujie2.gitbooks.io/python-introductory-and-advanced-data-analysis/chapter9.html>

3. 以及查找了其他关于 Python 读写 csv、xml 等内容

## 脚本文件说明

### getWords.py

使用 jieba 分词将训练数据集转化为词组

### getWordVec.py

将训练数据集词组结果转为向量

### getSVM.py

使用训练数据集训练支持向量机

### getWeibo.py

根据话题从微博爬取博文内容

### getPredict.py

对爬取到的微博数据进行预测

## 数据文件说明

### DataSet

爬取结果，同时也是预测所用到的测试集

### Result

预测结果

### SVM

通过训练集学习得到的支持向量机模型

### TrainingSet

训练集，以及一些中间产物。其中pos和neg分别表示积极和消极内容

ipad3 训练集来自于《2012年CCF自然语言处理与中文计算会议：中文微博情感分析测评数据》

<http://tcci.ccf.org.cn/conference/2012/pages/page10_dl.html>

普通训练集（neg.xls & pos.xls）来自于参考内容1，该语料库整合了书籍、计算机等 7个领域的评论数据。

## 须知

上述两个训练集都有比较大的局限性，因此训练出的支持向量机对于新数据的预测效果不尽如人意

因此上述训练数据集目前还只能用于参考和练习，并不能得出比较好的学习模型。
