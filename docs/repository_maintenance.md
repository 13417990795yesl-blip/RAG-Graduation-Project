# 仓库整理与发布记录

整理前主分支为 `cf1436b`，服务器仓库 `/data1/yesiliang/RAG` 工作区干净，已有5个主分支提交。完整Git历史和原跟踪文件已备份到服务器 `/data1/yesiliang/.codex-backups/RAG/20261015-160016/`。

## 整理内容

- 重写含无效shell片段的 `.gitignore`；保留真实 `config/.env`，仅提交空配置示例。
- 删除只有注释方案的 `src/rerank_utils2.py`；将 `fonts.py` 的字体检查合入绘图脚本。
- 把5份相同的KB解析循环合并至 `src/data_utils.py`，原函数名保留为兼容包装。
- 模型路径统一为项目相对路径，允许环境变量覆盖；修正旧中文脚本中不存在的小写 `rag` 绝对路径。
- 新增 `python -m src` 命令目录，原实验策略、候选池、collection和指标定义保持独立。
- 将调用LLM的 `test_multihoprag_llm_top20.py` 改名为 `diagnose_multihoprag_llm_top20.py`。
- 清空Notebook的21条运行输出和执行计数，保留14个单元的研究代码。
- MultiHop-RAG构建器增加 `--corpus`、`--qa`，默认读取项目数据目录，解除固定 `/tmp` 输入依赖。
- 完善README和直接依赖声明；绘图依赖单列为 `figures` 可选依赖。

服务器的虚拟环境、数据、模型、数据库、实验结果和字体文件均保留。缓存属于可重新生成的内容。原Notebook中的历史接口和路径未进行全流程迁移，README已标明其用途。

## 验证口径

检查语法、离线回归、命令分发和共享解析器在真实本地KB上的等价性。测试不调用收费LLM API，不重新嵌入语料或写入既有Chroma collection；本次整理不声称重现整套论文实验成绩。

详细机器执行记录保存在服务器 `results/repository_cleanup/`，不随源码上传。发布保留主分支历史，备份分支留在服务器；版本标签为 `repository-cleanup-v1`。

2026-10-15实际验证：9项离线回归通过；5个原解析器分别与ChronoQA（1,200块）、HotpotQA（500块）、MultiHop-RAG（4,884块）核对，共15组输出完全一致。129个原有函数的AST未改变；入口、KB解析包装和已记录的路径修正单独检查。命令列表、数据构建帮助和续跑帮助通过。字体检查在独立临时依赖环境中通过，原RAG虚拟环境未安装绘图包或升级模型依赖。

Git历史中的常见凭据模式检查、与真实配置密钥的精确匹配检查均未发现命中；真实 `.env` 未被修改。Notebook输出在当前版本清空，原始研究记录仍保留在原提交与服务器备份中。
