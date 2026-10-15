# RAG 重排序与生成质量实验

毕业设计《重排序模块在RAG系统中对生成质量的增效研究与实证分析》的实验代码，覆盖知识库构建、稠密检索、Pointwise、ColBERT、Listwise、Pairwise、SetSelection，以及答案生成和评估。

当前仓库保留原有实验历史，并整理了运行入口、路径配置和公共解析逻辑。数据集、模型权重、Chroma数据库、真实API配置和原始实验结果保留在服务器，不随源码上传。

## 安装与离线检查

使用Python 3.11和uv：

```bash
git clone https://github.com/13417990795yesl-blip/RAG-Graduation-Project.git
cd RAG-Graduation-Project
uv sync --frozen
cp config/.env.example config/.env
uv run python -m src --list
uv run python -m unittest discover -s tests -v
```

Windows PowerShell将复制命令改为 `Copy-Item config/.env.example config/.env`。检索与生成实验需要另外准备下述数据、模型；命令列表和离线测试不调用LLM API。

## 配置与目录

在 `config/.env` 填写自己的 `DEEPSEEK_API_KEY`。通过 `BGE_MODEL_PATH`、`BGE_RERANKER_PATH` 指定本地BGE-M3和bge-reranker-v2-m3目录；空值沿用 `models/BAAI/models--BAAI--.../snapshots/...` 的原始缓存路径。相对目录以项目根目录解析。

| 位置 | 用途 |
|---|---|
| `src/` | 数据构建、检索、重排、生成和共享工具 |
| `config/.env.example` | 可提交的空配置示例 |
| `tests/` | 无API调用的回归检查 |
| `notebook/rag-sys.ipynb` | 中文任务及LLM-as-Judge的历史探索记录，已清空输出 |
| `make_thesis_figures.py` | 论文配图及中文字体检查 |
| `thesis_figures/` | 原有论文配图 |
| `data/`、`models/`、`vector_db/` | 本地数据、权重与索引，不纳入Git |
| `results/`、`results_trimmed/` | 本地实验结果，不纳入Git |

Notebook的部分单元引用历史接口和原机器路径，保留用于研究过程查阅；新的MultiHop-RAG实验优先使用下面的命令入口。`src/rerank_utils.py` 是旧中文流程，导入时需要已准备好的模型、向量库和ColBERT缓存。

## 数据准备

| 数据 | 输入要求 | 构建入口 |
|---|---|---|
| ChronoQA | `data/chronoqa/chronoqa_source.json`，来源标识 `czy1999/ChronoQA` | `uv run python -m src build-chronoqa` |
| HotpotQA | 通过datasets加载 `hotpotqa/hotpot_qa` 的distractor validation | `uv run python -m src build-hotpotqa` |
| MultiHop-RAG | 原始 `corpus.json` 与 `MultiHopRAG.json` | 见下方命令 |

将MultiHop-RAG两个源文件放入 `data/multihoprag/`，或显式指定路径：

```bash
uv run python -m src build-multihoprag -- --corpus /path/to/corpus.json --qa /path/to/MultiHopRAG.json
uv run python -m src index-multihoprag
```

构建器沿用原来的50题、分块和随机种子设置。索引命令写入 `multihoprag_bge_collection_no_metadata`；旧的 `retrieve-multihoprag` 实验使用原metadata collection，两者保留独立口径。

## 重排与生成

以下命令需要准备好数据、模型和索引；LLM重排及生成会调用配置中的DeepSeek API。

```bash
# 支持续跑的Listwise / SetSelection重排
uv run python -m src rerank-multihoprag -- --sample-size 50 --methods listwise,setselection

# 为已有重排结果生成英文答案并计算生成指标
uv run python -m src generate-multihoprag

# GPU上的Baseline / Pointwise / ColBERT检索与答案生成
CUDA_VISIBLE_DEVICES=0 uv run python -m src compare-multihoprag -- --sample-size 50 --methods baseline,pointwise,colbert
```

统一入口只是转发已有实验，不合并不同候选池、collection和指标定义。旧的 `python src/脚本名.py` 方式仍可用于保留的脚本。原 `test_multihoprag_llm_top20.py` 改名为 `diagnose_multihoprag_llm_top20.py`，它是会调用API的实验诊断，不是单元测试。

## 论文配图

```bash
uv sync --frozen --extra figures
uv run --extra figures python make_thesis_figures.py --list-fonts
uv run --extra figures python make_thesis_figures.py
```

字体检查已合入绘图入口，原 `fonts.py` 已移除。绘图使用仓库原有字体和代码内的已记录数值，不会自动重跑RAG实验或刷新实验成绩。

清理范围、备份与验证记录见 [docs/repository_maintenance.md](docs/repository_maintenance.md)。
