# IMDb 评论情感分类综合实验

课程目标：公开数据爬虫采集 → 文本预处理 → 三种机器学习模型 → 科学评估 → 中文报告与交互演示。

本项目使用 **50,000 条有标签的英文电影评论**，对比多项式朴素贝叶斯、逻辑回归和线性 SVM，并增加多数类基线。结果由实际训练生成，清洗后数量和性能以 `artifacts/results/metrics.json` 为准。

本次全量运行保留 **49,578** 条评论（训练24,902，测试24,676），移除422条重复记录。

| 模型 | 训练CV Macro-F1 | 独立测试准确率 | 独立测试Macro-F1 |
|---|---:|---:|---:|
| 多项式朴素贝叶斯 | 87.90% | 87.19% | 87.18% |
| 逻辑回归 | 89.92% | 90.33% | 90.33% |
| 线性SVM | 90.07% | 90.11% | 90.11% |

按预设训练CV规则选择线性SVM。测试集上的逻辑回归准确率高0.22个百分点；报告保留这一差异，不以测试表现重新选择模型。验证记录见 `docs/verification.md`。

## 快速查看

- `reports/index.html`：可直接打开的中文网页报告、图表、误判示例。
- `reports/experiment_report.md`：完整中文实验报告。
- `docs/presentation_guide.md`：课堂演示顺序、模型讲解及答辩问题。
- `artifacts/results/model_comparison.csv`：量化对比表。
- `data/raw/provenance.json`：来源、采集时间、SHA-256 指纹与 robots 检查记录。

## 安装与运行

建议 Python 3.11+；本次实际环境为 Python 3.14。进入项目目录后：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
# 首次采集+清洗+训练+报告（约84 MB下载；无需账户）
sentiment-lab run
# 已经保留了数据包时完全离线重跑
sentiment-lab run --offline
```

需要严格复制本次依赖版本时，先 `python -m pip install -r requirements-lock.txt`，再安装本地包。不同 Python/操作系统可能无法使用同一版本的二进制 wheel；完整运行版本见 `metrics.json`。随机种子固定，耗时和浮点末位仍可能因硬件、库版本而不同。

按阶段运行或展示：

```bash
sentiment-lab collect       # 采集并校验原始评论，不做训练
sentiment-lab prepare --offline
sentiment-lab train
sentiment-lab report        # 仅重建图表及报告
sentiment-lab predict --text "The story was wonderful and the acting was excellent."
sentiment-lab serve         # http://127.0.0.1:8765，Ctrl+C停止
python -m pytest -q
```

无需安装本地包时可用 `PYTHONPATH=src python -m sentiment_lab run --offline`。从其他目录调用时：`sentiment-lab --root /absolute/project/path report`；全局参数放在子命令前。`--config` 可指定独立 JSON 配置；不应在观察测试集后调整配置并反复比较。

## 模块分层

```text
imdb_sentiment_lab/
├── .git/                    独立 Git 仓库
├── configs/                 实验配置
├── src/sentiment_lab/
│   ├── cli.py               命令解析与流程编排
│   ├── config.py / io.py    路径配置、序列化
│   ├── data/collect.py      HTML 链接发现、robots、分段下载、校验、归档解析
│   ├── data/prepare.py      清洗、去重、泄漏检查、数据审计
│   ├── models.py            TF-IDF Pipeline 与三种模型工厂
│   ├── train.py             参数搜索、训练集模型选择、持久化
│   ├── evaluation.py        指标、配对 bootstrap 区间
│   ├── visualization.py     PNG/SVG 图表
│   ├── report.py            基于真实结果生成中文报告
│   ├── predict.py           复用清洗和保存模型的推理
│   ├── server.py            本地展示服务
│   └── web/                 独立 HTML、CSS、JS 模板
├── tests/                   去重、泄漏、交叉验证及指标测试
├── data/raw/                来源页面、来源记录、本地缓存数据包
├── data/processed/          清洗后CSV（Git忽略，可重新生成）
├── artifacts/results/       数据审计、逐折CV、指标、预测、误判样例
├── artifacts/models/        保存的Pipeline（Git忽略，可重新训练）
├── reports/                 已生成的中文网页/Markdown报告及图表
└── docs/                    实验设计、讲解及检查记录
```

依赖方向：CLI → 数据/训练/报告；训练 → 模型工厂/评估；报告 → 指标产物/绘图；推理复用数据清洗及训练 Pipeline。CSS/JS 与 Python 分离，不把所有逻辑塞进一个文件。

## 实验原则

1. 官方原始训练/测试边界不混洗；空文本、同划分重复、同划分矛盾标签和跨划分重复有明确处理规则。
2. 只用正文特征，排除评分/文件名；TF-IDF 在各 CV 训练折内部拟合，避免词表泄漏。
3. 三种算法使用统一特征配置；分层 3 折训练 CV 搜索每种模型的两个参数值。
4. 按 CV macro-F1 选定模型并落盘，再查看独立测试表现；测试表现仅做最终方法对比。
5. 报告 Accuracy、Precision、Recall、F1、Macro-F1、ROC-AUC、混淆矩阵和耗时，给出准确率95%区间及配对差值区间。
6. 保存原始来源、数据指纹、配置、实际软件版本和逐条测试预测，便于复核。

## 数据采集边界

采集器真实请求 Stanford 公开发布页，从 HTML 自动提取下载链接，并下载原始数据包后解析其中的评论。它展示网页结构解析、robots 检查、请求超时、重试、分段下载、缓存及完整性验证能力。它不是直接逐页爬取 IMDb 的登录/动态评论页面；50,000 条评论由数据发布方先行采集和标注。缓存 SHA-256 用于本次采集的一致性检查，并非官方提供的签名。归档不解压到任意文件路径。

Git 忽略虚拟环境、原始84 MB压缩包、清洗后大文件和训练模型；本机保留这些内容。源码、配置、来源记录、量化结果、测试预测和报告纳入 Git。移动项目时需要连同忽略的数据/模型目录复制，或重新采集训练。

## 课程要求对照

| 要求 | 项目实现 |
|---|---|
| 数据量1W+ | 原始50,000有标签评论；清洗后数量由审计确认 |
| 三种以上模型 | NB、LR、Linear SVM，共三种；另有多数类基线 |
| 模型合理 | 稀疏文本特征，概率/判别/间隔三种方法，统一输入 |
| 计算科学 | 分层CV、Pipeline防泄漏、独立测试、固定seed |
| 量化分析 | 六项分类指标、混淆矩阵、CV差异、bootstrap、耗时 |
| 展示效果 | 中文网页、可缩放SVG、高清PNG、本地预测 |
| 实验讲解 | 完整报告、流程设计、课堂讲解与答辩指南 |
| 分层和Git | src模块、配置、测试、独立仓库及阶段提交 |

数据原始论文：Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL-HLT. https://aclanthology.org/P11-1015/
