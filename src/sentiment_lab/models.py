"""Model factory: identical TF-IDF settings, three distinct classifiers."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(name: str, config: dict) -> Pipeline:
    # 输入 name 为配置中的算法键，config 由 load_config 读取。
    # 返回未训练的 Pipeline；fit 接收 N 条文本和 N 个标签，predict 返回 N 个 0/1 标签。
    # 文本经过 tfidf 变为形状 (评论数 N, 词表大小 V) 的矩阵，再交给 clf。
    # 每行只有本条评论中出现的特征非零，采用稀疏表示可避免分配完整 N×V 数组。
    # 工厂只构造未拟合模型；所有算法共用下面的 TF-IDF 设置，控制比较中的特征变量。
    # NB 估计类别条件下的词语倾向；LR 学习正则化线性权重；SVM 优化线性分类间隔。
    # LR/SVM 的随机种子固定，迭代上限提高到适合当前稀疏文本任务的水平。
    # NB 的 alpha 是平滑强度，避免未出现词的条件概率为零；在配置网格中搜索。
    # LR/SVM 的 C 控制正则化，C 越小约束越强，C 越大越重视拟合训练数据。
    # liblinear 适合当前二分类任务；LinearSVC 的 dual='auto' 按问题形状等选择求解形式。
    # max_iter 是优化迭代上限，并非对数据重复训练的次数；不收敛由训练模块显式报错。
    classifiers = {
        "multinomial_nb": MultinomialNB(),
        "logistic_regression": LogisticRegression(solver="liblinear", max_iter=2000, random_state=config["seed"]),
        "linear_svm": LinearSVC(dual="auto", max_iter=5000, random_state=config["seed"]),
    }
    # 复制配置后转换 JSON 数组为 sklearn 使用的元组，避免原地修改实验配置。
    options = dict(config["features"])
    options["ngram_range"] = tuple(options["ngram_range"])
    # Retain negations (no English stop-word list). Float32 keeps sparse features compact.
    # TF-IDF 根据文本内词频与跨文档频率赋权：出现于很多文档的词通常区分度较低。
    # 默认 1–2 gram 同时表示单词与 not good 等相邻词组；max_features 限制词表规模，
    # min_df/max_df 过滤过少/过多文档中出现的词，sublinear_tf 用 1+log(tf) 缓和高词频。
    # 不配置英文停用词表以保留否定词；float32 减少稀疏特征矩阵的内存占用。
    # 以 'not good movie' 为例，1–2 gram 候选包含 not、good、movie、not good、good movie；
    # 候选还要通过文档频率和词表上限筛选，并不保证每个词组都会进入最终词表。
    # min_df=3 指至少在当前拟合语料的 3 篇文档出现，而非在一篇中重复出现 3 次；
    # max_df=0.98 指排除覆盖超过 98% 文档的特征。数字均来自默认实验配置。
    # sklearn 默认平滑 IDF 为 log((1+N)/(1+df))+1，df 是含该特征的文档数；
    # 与处理后的 TF 相乘，再按默认 L2 范数归一化每行，缓和长评论的整体长度影响。
    # 词表中的列索引在训练后固定，推理时未知词被忽略，已知词仍映射到相同列。
    features = TfidfVectorizer(dtype=np.float32, **options)
    # 关键防泄漏边界：GridSearchCV 对每一折重新拟合整个 Pipeline，词表与 IDF
    # 只来自当前训练折。若先对全量文本 fit_transform，再做 CV，就会泄漏验证信息。
    # clf 是步骤名，因此参数网格用 clf__alpha / clf__C 定位分类器参数。
    # fit 时 tfidf.fit_transform → clf.fit；predict 时 tfidf.transform → clf.predict。
    # 训练阶段保存这个返回对象的已拟合版本，避免只保存分类器却丢失词表和 IDF。
    return Pipeline([("tfidf", features), ("clf", classifiers[name])])
