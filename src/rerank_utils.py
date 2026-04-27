import os
from typing import List
import chromadb
import pickle
import torch
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import re
import math
from typing import List, Tuple, Dict, Any
from FlagEmbedding import FlagReranker, BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import json
# 全局加载缓存（只加载一次）
with open('/data1/yesiliang/rag/vector_db/colbert_multifieldqa/colbert_cache.pkl', 'rb') as f:
    COLBERT_CACHE = pickle.load(f)
    
load_dotenv("/data1/yesiliang/rag/config/.env")

# 客户端初始化
# 如果想保存到磁盘，建议使用 PersistentClient:
#client = chromadb.PersistentClient(path="/data1/yesiliang/rag/vector_db/multifieldqa")
#client = chromadb.PersistentClient(path="/data1/yesiliang/rag/vector_db/contextual_multifieldqa") 
client = chromadb.PersistentClient(path="/data1/yesiliang/rag/vector_db/dureader")
# 获取或创建集合
#collection_name = "baai_bge_collection"
#collection_name = "contextual_collection"
collection_name = "dureader_collection"
collection = client.get_or_create_collection(name=collection_name)

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# 初始化模型
BASE_MODEL_DIR = '/data1/yesiliang/rag/models/BAAI/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
RERANKER_DIR = '/data1/yesiliang/rag/models/BAAI/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'

embedding_model = SentenceTransformer(BASE_MODEL_DIR)
colbert_model = BGEM3FlagModel(BASE_MODEL_DIR, use_fp16=True)
reranker_model = FlagReranker(RERANKER_DIR, use_fp16=True)


def embed_chunk(chunk: str) -> List[float]:
    # BGE 模型在计算时建议开启 normalize_embeddings=True 以获得更好的余弦相似度结果
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()
    
#批量处理建议：不要一个一个循环 embed_chunk，直接传 List 效率更高
def embed_chunks_batch(chunks: List[str]) -> List[List[float]]:
    # 这里的 batch_size 可以根据显存调整
    embeddings = embedding_model.encode(chunks, batch_size=12, normalize_embeddings=True)
    return embeddings.tolist()

def save_embeddings(ids: List[str], chunks: List[str], embeddings: List[List[float]]) -> None:
    """
    将文本块及其对应的向量批量存入 ChromaDB
    """
    # ChromaDB 的 .add 方法本身就支持 List。
    # 你的 embeddings 本身已经是 List[List[float]] (逗号分隔的浮点数列表)
    # 这正是 ChromaDB 所需要的格式。
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print(f"成功存入 {len(chunks)} 条向量数据。")

def parse_kb_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按照 --- 分割块
    raw_chunks = content.split('---')
    
    final_chunks = []
    final_ids = []
    
    for c in raw_chunks:
        c = c.strip()
        if not c: continue
        
        # 使用正则表达式提取 [题_x_块_y]
        match = re.search(r'\[(题_\d+_块_\d+)\]', c)
        if match:
            chunk_id = match.group(1) # 提取出 "题_0_块_0"
            # 移除 ID 标签，只保留纯文本内容（可选，但建议保留纯净文本）
            text = re.sub(r'\[题_\d+_块_\d+\]', '', c).strip()
            
            final_ids.append(chunk_id)
            final_chunks.append(text)
            
    return final_ids, final_chunks

def save_enhanced_kb_to_disk(enhanced_data, output_path="longbench_kb_contextual.md"):
    """
    将增强后的数据按照原始 longbench_kb.md 的格式保存到磁盘。
    
    enhanced_data 格式预期: [{"id": "题_0_块_0", "text": "[Context: ...] 正文"}, ...]
    """
    formatted_blocks = []
    
    for item in enhanced_data:
        # 构造每一个块的字符串：[ID] 内容
        block_content = f"[{item['id']}] {item['text']}"
        formatted_blocks.append(block_content)

    # 使用原始的 --- 分隔符将所有块连接起来
    # 并在分隔符前后保留换行，确保格式整齐
    full_content = "\n\n---\n\n".join(formatted_blocks)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f"✅ 增强版知识库已保存至: {output_path}")
    print(f"总计保存块数量: {len(enhanced_data)}")



# def precompute():
#     # 1. 从你之前的逻辑中获取所有的 ids 和 chunks
#     # 假设你已经有了 real_ids 和 real_chunks (共 481 条)
    
#     print("正在计算 481 个块的 ColBERT 向量...")
#     # 批量计算（建议根据显存调整 batch_size）
#     # 返回的是 list of tensors
#     outputs = model.encode(chunks, return_colbert_vecs=True)
#     colbert_vecs = outputs['colbert_vecs']
    
#     # 2. 建立 ID -> Vector 的映射字典
#     # 注意：BGE-M3 的 colbert_vecs 是 list of numpy arrays
#     cache = {chunk_id: vec for chunk_id, vec in zip(real_ids, colbert_vecs)}
    
#     # 3. 保存到本地
#     with open('colbert_cache.pkl', 'wb') as f:
#         pickle.dump(cache, f)
#     print("预计算完成，已保存至 colbert_cache.pkl")

def colbert_rerank(query: str, initial_ids: List[str], initial_chunks: List[str]) -> Tuple[List[str], List[str]]:
    if not initial_ids:
        return [], []

    q_reps = colbert_model.encode([query], return_colbert_vecs=True)
    q_vec = np.asarray(q_reps['colbert_vecs'][0])  # shape: [Lq, D]

    scores = []
    for cid in initial_ids:
        d_vec = np.asarray(COLBERT_CACHE[cid])      # shape: [Ld, D]
        sim = q_vec @ d_vec.T                       # [Lq, Ld]
        score = sim.max(axis=1).sum() / q_vec.shape[0]
        scores.append(score)

    order = np.argsort(scores)[::-1]
    final_ids    = [initial_ids[i]     for i in order]
    final_chunks = [initial_chunks[i]  for i in order]
    return final_ids, final_chunks

def retrieve(query: str, top_k: int):
    # 1. 为 BAAI 模型准备查询指令 (这是提升精度的关键)
    # 如果你使用的是 BGE-M3 或 BGE-Large-zh
    instruction = "为这个句子生成表示以用于检索相关文章："
    
    # 2. 向量化查询词 (注意这里加了前缀)
    # 这里建议直接使用 embedding_model.encode
    query_embedding = embedding_model.encode(
        instruction + query, 
        normalize_embeddings=True,
        show_progress_bar=False 
    ).tolist()
    
    # 3. 在 ChromaDB 中检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"] 
    )

    ids = results['ids'][0]
    
    # 提取文档
    documents = results['documents'][0]
    
    # 提取距离并转化为相似度分值 (1 - 距离 = 相似度)
    # Chroma 默认返回的是余弦距离，0 代表完全一致，1 代表完全不相关
    scores = [1.0 - d for d in results['distances'][0]]
    
    return ids, documents, scores

# def generate(query: str, chunks: List[str]) -> str:
#     # 1. 增加安全检查：如果没有检索到任何片段
#     if not chunks:
#         return "抱歉，在现有的知识库中没有找到相关信息来回答这个问题。"

#     chunks_text = "\n\n".join([f"【片段{i+1}】: {chunk}" for i, chunk in enumerate(chunks)])
    
#     # 2. 优化 Prompt：增加逻辑限定
#     prompt = f"""你是一位专业的知识助手。请基于下方提供的“相关片段”来回答“问题”。

# ### 问题: 
# {query}

# ### 相关片段:
# {chunks_text}

# ### 要求:
# 请直接给出答案，不要有任何多余的解释，尽量简短。

# ### 回答：
# """

#     try:
#         response = deepseek_client.chat.completions.create(
#             model="deepseek-chat",
#             messages=[
#                 {"role": "system", "content": "你是一位专业的知识助手，擅长从给定文档中提取准确信息。"},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,  # 低温度：增加回答的确定性和准确性，减少随机性
#             stream=False      # 如果需要流式输出，请改为 True 并修改返回逻辑
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"生成答案时出错: {str(e)}"

def generate(query: str, chunks: List[str]) -> str:
    if not chunks:
        return "抱歉，在现有的知识库中没有找到相关信息来回答这个问题。"

    # 拼接上下文（即检索回来的片段）
    context = "\n\n".join(chunks)
    
    prompt = f"""请基于下面“文档”中的内容回答问题。

# ###问题：
#{query}

# ###文档：
#{context}

# ###要求：
#请直接给出答案。不要输出任何多余的解释或寒暄。尽量简洁。

# ###答案：
#"""
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成答案时出错: {str(e)}"

def deepseek_listwise_rerank(query: str, initial_ids: List[str], initial_chunks: List[str]) -> Tuple[List[str], List[str]]:
    """
    使用 DeepSeek 对 20 个片段进行全量 Listwise 排序。
    返回: (排序后的ID列表, 排序后的文本列表)
    """
    if not initial_chunks:
        return [], []

    # 1. 构造带有临时编号(0-19)的片段供模型识别
    # 我们需要记录一个映射关系：临时编号 -> 原始ID 和 原始文本
    context_entries = []
    for i, chunk in enumerate(initial_chunks):
        context_entries.append(f"--- 片段 ID: {i} ---\n{chunk}")

    chunks_text_for_prompt = "\n\n".join(context_entries)

    # 2. 构建针对“排序”的 Prompt
    rerank_prompt = f"""你是一个智能搜索排序专家。请对比以下 20 个候选片段与用户问题的相关性。

### 用户问题: 
{query}

### 候选片段列表:
{chunks_text_for_prompt}

### 任务要求:
1. 深度分析每个片段是否包含回答用户问题的核心事实、数据或逻辑。
2. 请按照相关性从高到低，对这 20 个片段的 ID 进行排序。
3. 你的输出格式必须是：以英文逗号分隔的数字序列（例如：5,2,19,0...）。
4. **严禁** 输出任何解释、前言或除了数字和逗号以外的字符。
5. 必须包含所有输入的 20 个 ID，不要遗漏。

请输出排序结果:"""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个只输出数字列表的排序机器人。"},
                {"role": "user", "content": rerank_prompt}
            ],
            temperature=0.1, 
            stream=False
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # 3. 解析模型返回的顺序
        # 提取所有数字
        predicted_order = [int(num) for num in re.findall(r'\d+', raw_output)]
        
        # 4. 安全性处理：如果模型漏掉了某些ID或返回了错误的ID
        seen = set()
        final_order = []
        for idx in predicted_order:
            if idx < len(initial_chunks) and idx not in seen:
                final_order.append(idx)
                seen.add(idx)
        
        # 补全没出现在预测列表里的原始ID（保持原有初筛顺序）
        for i in range(len(initial_chunks)):
            if i not in seen:
                final_order.append(i)

        # 5. 根据 final_order 重组 ID 和 文本
        ranked_ids = [initial_ids[i] for i in final_order]
        ranked_chunks = [initial_chunks[i] for i in final_order]
        
        return ranked_ids, ranked_chunks
        
    except Exception as e:
        print(f"DeepSeek Listwise 重排出错: {e}")
        # 出错时保底：返回初筛的原样顺序
        return initial_ids, initial_chunks

def deepseek_pairwise_rerank(query: str, initial_ids: List[str], initial_chunks: List[str]) -> Tuple[List[str], List[str]]:
    """
    使用锦标赛对决法对 20 个片段进行排序，返回全量排序后的 ID 和文本列表。
    """
    if not initial_chunks:
        return [], []

    # 包装候选人，记录原始信息和被淘汰的轮次
    # defeated_by: 记录击败该候选人的对手索引
    candidates = []
    for i in range(len(initial_chunks)):
        candidates.append({
            "temp_idx": i,
            "id": initial_ids[i],
            "text": initial_chunks[i],
            "eliminated_round": 99, # 默认为冠军（未被淘汰）
            "defeated_by": -1
        })

    current_round_candidates = candidates.copy()
    round_num = 1
    
    # --- 阶段 1: 锦标赛对决 ---
    while len(current_round_candidates) > 1:
        pairs = []
        for i in range(0, len(current_round_candidates), 2):
            if i + 1 < len(current_round_candidates):
                pairs.append((current_round_candidates[i], current_round_candidates[i+1]))
            else:
                pairs.append((current_round_candidates[i], None)) # 轮空

        # 构造批量对决指令
        matches_text = ""
        for midx, (cand_a, cand_b) in enumerate(pairs):
            if cand_b:
                matches_text += f"【对决组 {midx}】\n文档 A: {cand_a}\n文档 B: {cand_b}\n\n"

        if matches_text:
            prompt = f"""你是一位专业的搜索排序专家。请针对用户问题，在以下几组文档对决中选出相关度更高的一项。

### 用户问题: 
{query}

### 对决列表:
{matches_text}

### 任务要求:
1. 针对每一组对决，只能选择 A 或 B。
2. 必须严格按照此格式输出：对决组序号: 胜者字母（例如 0: A, 1: B）。不要有任何解释。

结果:"""
            try:
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                res_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"API 调用失败: {e}")
                res_text = ""

            # 处理胜负结果
            next_round = []
            for midx, (cand_a, cand_b) in enumerate(pairs):
                if cand_b is None:
                    next_round.append(cand_a)
                    continue
                
                # 正则匹配胜者
                win_match = re.search(rf"{midx}:\s*([A-B])", res_text, re.IGNORECASE)
                if win_match and win_match.group(1).upper() == "B":
                    # B 胜
                    cand_a['eliminated_round'] = round_num
                    cand_a['defeated_by'] = cand_b['temp_idx']
                    next_round.append(cand_b)
                else:
                    # A 胜 (含解析失败保底)
                    cand_b['eliminated_round'] = round_num
                    cand_b['defeated_by'] = cand_a['temp_idx']
                    next_round.append(cand_a)
            
            current_round_candidates = next_round
            round_num += 1
        else:
            break

    # 锦标赛冠军
    winner = current_round_candidates[0]

    # --- 阶段 2: 修正 Top-2 (可选，但推荐) ---
    # 真正的第二名一定在输给冠军的那些人里面
    victims_of_winner = [c for c in candidates if c['defeated_by'] == winner['temp_idx']]
    if len(victims_of_winner) > 1:
        # 简单做一次列表排序确定谁是老二
        # 这里为了省 API，可以直接按他们原本坚持的轮次降序排，轮次越高越强
        victims_of_winner.sort(key=lambda x: x['eliminated_round'], reverse=True)
        # 将 victims_of_winner[0] 标记为比其他被淘汰者更高的优先级
        victims_of_winner[0]['eliminated_round'] = round_num - 0.5 

    # --- 阶段 3: 构造全量列表 ---
    # 排序规则：淘汰轮次越晚，排名越靠前
    final_sorted_cands = sorted(candidates, key=lambda x: x['eliminated_round'], reverse=True)
    
    ranked_ids = [c['id'] for c in final_sorted_cands]
    ranked_chunks = [c['text'] for c in final_sorted_cands]

    return ranked_ids, ranked_chunks

def bge_pointwise_rerank(query: str, ids: List[str], chunks: List[str]) -> Tuple[List[str], List[str], List[float]]:
    """
    输入：
        query: 用户问题
        ids: 初筛返回的 ID 列表
        chunks: 初筛返回的文本块内容列表
        top_k: 重排后保留的前几个结果
    输出：
        Tuple: (排序后的ID列表, 排序后的文本列表, 对应的分数)
    """
    if not chunks:
        return [], [], []
        
    pairs = [[query, chunk] for chunk in chunks]
    
    # 2. 计算相关性得分 (分数越高越相关)
    # BGE Reranker 返回的是 float 分数
    scores = reranker_model.compute_score(pairs)
    
    # 3. 将 ID、内容、分数打包在一起
    # zip 会按照索引一一对应：(id_0, chunk_0, score_0), (id_1, chunk_1, score_1)...
    combined = list(zip(ids, chunks, scores))
    
    # 4. 按分数从高到低排序 (x[2] 是 score)
    ranked_combined = sorted(combined, key=lambda x: x[2], reverse=True)
    
    # 6. 解包并返回
    # zip(*...) 是 zip 的逆操作，将元组列表转回多个列表
    final_ids, final_chunks, final_scores = zip(*ranked_combined)
    
    return list(final_ids), list(final_chunks), list(final_scores)

TEST_VAR = 43

def get_chunks_by_question_id(qid: int) -> List[str]:
    chunks = []
    with open("dureader_kb.md", "r", encoding="utf-8") as f:
        content = f.read()

    # 按 --- 分割
    raw_blocks = content.split("---")
    prefix = f"[题_{qid}_块_"

    for block in raw_blocks:
        block = block.strip()
        if prefix in block:
            # 去掉 [题_x_块_y] 标签，只留文本
            text = re.sub(r'\[.*?\]', '', block).strip()
            if text:
                chunks.append(text)

    return chunks

def parse_setselection_output(raw_output: str, total_chunks: int, top_k: int) -> List[int]:
    """
    解析 DeepSeek 返回的 SetSelection 输出，提取合法的块编号
    """
    # 提取所有数字
    numbers = re.findall(r'\d+', raw_output)
    indices = []
    seen = set()
    
    for num_str in numbers:
        idx = int(num_str)
        if 0 <= idx < total_chunks and idx not in seen:
            indices.append(idx)
            seen.add(idx)
            if len(indices) == top_k:
                break
    
    # 如果不够 top_k 个，用未选中的块补全
    if len(indices) < top_k:
        remaining = [i for i in range(total_chunks) if i not in seen]
        indices.extend(remaining[:top_k - len(indices)])
    
    return indices[:top_k]

def deepseek_setselection(question: str, chunks: List[str], top_k: int = 5) -> List[str]:
    """
    使用 DeepSeek API 实现 SetSelection 架构：
    从候选块集合中一次性挑选出最相关的 top_k 个块
    
    参数:
        question: 问题文本
        chunks: 候选文本块列表
        top_k: 需要挑选的块数
    
    返回:
        被选中的文本块列表（按相关性排序）
    """
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        return chunks
    
    # 1. 为每个块编号
    numbered_chunks = []
    for i, chunk in enumerate(chunks):
        numbered_chunks.append(f"[块{i}] {chunk}")
    
    chunks_text = "\n\n".join(numbered_chunks)
    
    # 2. 构建 SetSelection Prompt
    prompt = f"""# 任务：从下方候选文本块中，挑选出对回答指定问题最有帮助的 {top_k} 个块。

# 要求：
1. 仔细阅读每个文本块的内容。
2. 判断哪些块包含了回答问题所需的关键信息。
3. 只输出被选中块的编号列表，用逗号分隔，例如: 0, 3, 7, 12, ...
4. 必须输出恰好 {top_k} 个编号，不要多也不要少。
5. 按你认为的相关性从高到低排序。
6. 不要输出任何其他内容，只输出编号列表。

# 问题：{question}

# 候选文本块：
{chunks_text}

# 被选中的 {top_k} 个块的编号（有序，逗号分隔）："""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "你是一个专业的文本检索助手，擅长从文档集合中精准挑选出与问题最相关的片段。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 极低温度，确保输出稳定
            stream=False
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # 3. 解析模型输出的编号
        selected_indices = parse_setselection_output(raw_output, len(chunks), top_k)
        
        # 4. 返回被选中的块
        selected_chunks = [chunks[idx] for idx in selected_indices]
        return selected_chunks
        
    except Exception as e:
        import traceback
        print(f"SetSelection 出错: {e}")
        traceback.print_exc()  # 打印完整的调用栈
        return chunks[:top_k]

