# 项目协作指南

本文件适用于整个仓库。修改前先阅读 `README.md`，涉及实验方法时同时阅读 `docs/experiment_design.md`。默认使用中文沟通和编写面向课程的文档，代码标识符沿用现有英文命名。

## 项目目标与技术栈

- 本项目是 IMDb 英文电影评论二分类综合实验：公开数据采集 → 清洗审计 → TF-IDF → 三种分类器与多数类基线 → 科学评估 → 中文报告与交互演示。
- Python >= 3.11，采用 `src/` 包布局和 setuptools；依赖声明在 `pyproject.toml`，已记录的运行环境版本在 `requirements-lock.txt`。
- 三种分类器为 MultinomialNB、LogisticRegression、LinearSVC；主要依赖 scikit-learn、pandas、NumPy、Matplotlib、requests、BeautifulSoup 和 joblib。
- 前端为独立 HTML/CSS/JavaScript，无需 npm 构建；本地演示服务使用 Python 标准库。

## 模块与修改位置

| 路径 | 职责 |
| --- | --- |
| `src/sentiment_lab/cli.py`、`__main__.py` | 命令入口与阶段编排，保持命令层简洁 |
| `src/sentiment_lab/config.py`、`io.py` | 项目路径、配置读取与 JSON 序列化 |
| `src/sentiment_lab/data/collect.py` | 发布页解析、robots 检查、下载重试与完整性验证、归档读取、来源记录 |
| `src/sentiment_lab/data/prepare.py` | 统一清洗、重复与冲突处理、跨划分泄漏检查、数据审计 |
| `src/sentiment_lab/models.py` | 统一 TF-IDF Pipeline 与分类器工厂 |
| `src/sentiment_lab/train.py`、`evaluation.py` | 训练 CV 搜索、模型选择、持久化、测试指标与配对 bootstrap |
| `src/sentiment_lab/report.py`、`visualization.py` | 从实际实验产物生成中文报告及 PNG/SVG 图表 |
| `src/sentiment_lab/predict.py`、`server.py` | 已保存模型推理、仅本机访问的展示服务 |
| `src/sentiment_lab/web/` | 报告网页模板、CSS、JavaScript 的源文件 |
| `configs/experiment.json` | 随机种子、特征设置、参数网格与资源配置 |
| `tests/` | 清洗、下载、模型 Pipeline、指标与 bootstrap 测试 |
| `docs/` | 实验设计、演示说明、历史验证记录 |

依赖方向保持为 CLI → 数据/训练/报告，训练 → 模型/评估，报告 → 结果/绘图；推理复用清洗函数和训练 Pipeline。遵循现有模块风格，避免把业务逻辑集中到 CLI 或网页模板。

## 安装与常用命令

在项目根目录执行，优先复用已有 `.venv`；尚未创建环境时：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

需要复现记录的依赖版本时，先安装 `requirements-lock.txt`，再安装本地包；注意不同 Python/操作系统的 wheel 可用性。

```bash
python -m pytest -q
sentiment-lab collect
sentiment-lab prepare --offline
sentiment-lab train
sentiment-lab report
sentiment-lab predict --text "The story was wonderful and the acting was excellent."
sentiment-lab serve
sentiment-lab run --offline
```

- `run` 依次采集、清洗、训练、生成报告，会覆盖相关实验产物；联网运行还可能下载约 84 MB 数据。根据修改范围选择阶段，不为普通文档或样式修改重跑完整实验。
- `--offline` 适用于 `collect`、`prepare`、`run`，需要本地原始归档和采集记录，缓存缺失时不能保证离线执行成功。
- 未安装本地包时可使用 `PYTHONPATH=src python -m sentiment_lab <子命令>`。
- 全局 `--root`、`--config` 必须放在子命令前，例如 `sentiment-lab --root /absolute/project/path report`。
- `serve` 默认地址为 `http://127.0.0.1:8765`，需要已生成报告和已保存模型。

## 必须保持的实验约束

- 保持原始 train/test 边界；不把测试评论移入训练，不用测试表现反复调整参数或重新选择模型。
- 只用评论正文作为特征，不引入评分、文件名或其他标签线索；标签 0 为负面、1 为正面。
- 清洗去除 HTML、解码实体、统一小写与空白，并保留否定词；训练和推理必须使用同一清洗逻辑。
- 按规范化正文 SHA-256 去重：同划分矛盾标签全部剔除，同划分重复保留一条，跨划分重复只从测试集移除。保留每一步的审计数量。
- TF-IDF 必须放在 Pipeline 内，在各 CV 训练折拟合词表和 IDF；三种分类器使用统一特征配置。
- 用固定随机种子的分层 CV 和训练 CV macro-F1 选择模型；先写入 `selection.json`，再生成独立测试预测。保留多数类基线。
- 默认配置为 seed=42、3 折 CV、`n_jobs=1`、1000 次 bootstrap；配置由 `configs/experiment.json` 管理，避免散落硬编码。保持稀疏 float32 特征和训练线程限制，避免无必要的内存开销。
- 混淆矩阵行是真实标签、列是预测标签，标签顺序为 `[0, 1]`；ROC-AUC 使用正面概率或决策分数，决策分数不能称为概率。
- Bootstrap 模型比较使用相同重采样索引；收敛失败应报错，不能隐藏警告后生成成功结论。
- 结果、图表和结论必须来自真实运行产物，不能编造指标。当前选中模型、样本数量和性能以 `artifacts/results/` 中的记录为准，不写死历史结果。

## 源文件与生成产物

- 网页修改应落在 `src/sentiment_lab/web/`，报告文本修改应落在 `report.py`，图表修改应落在 `visualization.py`；随后按需执行 `sentiment-lab report` 同步生成文件。
- `reports/index.html`、`reports/style.css`、`reports/app.js`、`reports/experiment_report.md` 和 `reports/figures/` 是生成产物，避免仅修改这些文件而让下次生成覆盖变更。
- `artifacts/results/` 保存审计、逐折 CV、模型选择、配置快照、指标、测试预测和误判示例，当前纳入 Git；保留产物间的一致性，不能手改数值来修正实验结果。
- `.gitignore` 排除了 `.venv/`、原始压缩包、清洗数据、模型及缓存。不要为了提交而把这些大文件加入版本管理，也不要无故删除本地缓存或模型。
- 采集来自 Stanford 的公开发布页和原始数据包。描述中应明确采集方式；本地 SHA-256 是采集一致性指纹，不是官方签名。
- 下载逻辑应保持超时、重试、范围/长度与缓存校验；归档按成员读取，不向任意路径解压。
- 仅加载本项目可信训练输出的 joblib 模型；本地服务保持 loopback 绑定、输入长度检查、已知英文特征检查和 Origin 校验。

## 验证与交付

- 文档修改核对路径、命令与源码是否一致，无需重跑训练。
- Python 行为修改运行相关 pytest 测试；涉及清洗、模型、评估或下载公共逻辑时运行 `python -m pytest -q`。测试使用小型样本、临时目录和网络替身，避免依赖真实下载或全量训练。
- Python 修改可补充 `python -m compileall -q src`；JavaScript 修改运行 `node --check src/sentiment_lab/web/app.js`（需要 Node.js）。仓库未配置独立 lint 或类型检查工具，不假设存在这些命令。
- 报告或前端修改后，使用已有结果生成报告，检查占位符、图表、误判筛选与布局；预测相关修改验证正负英文输入、空文本、无已知特征文本和超长输入。
- 方法或配置变更应同步实验设计和说明；重新训练后检查数据审计、CV 选择、逐条预测、指标与报告的一致性，再更新相关文档中的结果。
- `docs/verification.md` 是历史验证记录，不能将其中的成功检查当成本次已经执行；新增记录只写实际运行的命令与观察。
- 交付说明写清修改内容、实际验证及未完成检查。保留已有用户改动，避免无关重构、依赖升级和生成文件噪声。
