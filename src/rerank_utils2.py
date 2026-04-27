import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# def adaptive_evaluate_logic(query, i, pbar, threshold=0.15):
#     """
#     自适应重排序执行逻辑
#     threshold: 置信度分差阈值，建议设在 0.1 到 0.2 之间
#     """
#     # 1. 粗排：拿 10 个候选，并获取它们的相似度得分
#     initial_chunks, scores = retrieve(query, 10)
    
#     # 2. 计算置信度分差 (Confidence Gap)
#     # 取第一名和第二名的分差
#     if len(scores) >= 2:
#         confidence_gap = scores[0] - scores[1]
#     else:
#         confidence_gap = 1.0 # 只有一个结果时，置信度最高

#     # 3. 门控决策
#     if confidence_gap > threshold:
#         # 情况 A：分差很大，初筛非常精准，直接跳过重排
#         if pbar:
#             pbar.set_description(f"问题 {i} | 跳过重排 (分差:{confidence_gap:.3f})")
#         final_chunks = [initial_chunks[0]] # 取第一名
#     else:
#         # 情况 B：分差很小，候选者之间存在语义模糊，触发重排
#         if pbar:
#             pbar.set_description(f"问题 {i} | 触发重排 (分差:{confidence_gap:.3f})")
        
#         # 这里调用你选定的重排函数（如 BGE 或 DeepSeek）       
#         # 2.1 bge-pointwise。
#         final_chunks = rerank_chunks(query, initial_chunks, top_k=1)
            
#         # 2.5 使用 DeepSeek 进行序列重排 (Listwise)
#         #final_chunks = deepseek_rerank(query, initial_chunks)
            
#         # 2.6 锦标赛重排（从 10 个里选出唯一的王者）
#         # 注意：这会产生 9 次 API 调用
#         #final_chunks = deepseek_pairwise_tournament(query, initial_chunks, pbar=pbar, q_idx=i)
        
#     return final_chunks

# def mmr_listwise_rerank(query: str, top_n_initial=30, top_k_mmr=10, lambda_param=0.5, pbar=None, q_idx=0):
#     """
#     一体化重排序逻辑：
#     1. 初筛检索 (Top 30) 并直接获取向量
#     2. MMR 算法选出相关且多样的片段 (Top 10)
#     3. DeepSeek Listwise 选出最后的唯一冠军 (Top 1)
#     """
#     # --- 1. 向量化提问 (仅执行 1 次) ---
#     instruction = "为这个句子生成表示以用于检索相关文章："
#     query_embedding_list = embedding_model.encode(instruction + query, normalize_embeddings=True).tolist()
#     query_vec = np.array(query_embedding_list).reshape(1, -1)

#     # --- 2. 在 ChromaDB 中检索 Top-N 片段及其向量 ---
#     # 关键修改：include 增加了 "embeddings"
#     results = collection.query(
#         query_embeddings=[query_embedding_list],
#         n_results=top_n_initial,
#         include=["documents", "distances", "embeddings"] 
#     )

#     all_chunks = results['documents'][0]
#     all_embeddings = np.array(results['embeddings'][0]) # 直接拿到数据库里的向量，不重复计算
#     all_scores = [1.0 - d for d in results['distances'][0]] # 转化为相似度

#     if not all_chunks: return []

#     # --- 3. 执行 MMR 多样化筛选 (从 30 个里选 10 个) ---
#     if pbar: pbar.set_description(f"问题 {q_idx} | 正在进行 MMR 多样化去重...")
    
#     selected_indices = []
#     unselected_indices = list(range(len(all_chunks)))
    
#     # 先选出最相关的一个作为起点
#     first_idx = np.argmax(all_scores)
#     selected_indices.append(first_idx)
#     unselected_indices.remove(first_idx)

#     while len(selected_indices) < top_k_mmr and unselected_indices:
#         mmr_scores = []
#         for idx in unselected_indices:
#             # 相关性：该片段与问题的相似度
#             relevance = all_scores[idx]
#             # 冗余性：该片段与已选片段的最大相似度
#             target_vec = all_embeddings[idx].reshape(1, -1)
#             selected_vecs = all_embeddings[selected_indices]
#             redundancy = np.max(cosine_similarity(target_vec, selected_vecs))
            
#             # MMR 公式
#             score = lambda_param * relevance - (1 - lambda_param) * redundancy
#             mmr_scores.append(score)
        
#         best_candidate_idx = unselected_indices[np.argmax(mmr_scores)]
#         selected_indices.append(best_candidate_idx)
#         unselected_indices.remove(best_candidate_idx)

#     # 得到 10 个高密度、去重后的片段
#     diverse_chunks = [all_chunks[i] for i in selected_indices]

#     # --- 4. DeepSeek Listwise 决赛 (从 10 个里选 1 个) ---
#     if pbar: pbar.set_description(f"问题 {q_idx} | MMR 完成 -> DeepSeek 深度重排中...")
    
#     # 调用你之前写的 deepseek_rerank 函数
#     # 注意：diverse_chunks 已经是 List[str] 格式
#     final_winner_list = deepseek_rerank(query, diverse_chunks) 

#     return final_winner_list

