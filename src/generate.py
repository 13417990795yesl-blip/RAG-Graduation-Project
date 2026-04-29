import json
import time
from tqdm import tqdm # 如果没装，执行 pip install tqdm
from src.rerank_utils import retrieve, bge_pointwise_rerank, colbert_rerank, deepseek_listwise_rerank, deepseek_pairwise_rerank, generate, get_chunks_by_question_id, deepseek_setselection
import numpy as np
import re
import string
from collections import Counter

def evaluate_rag_system(test_questions, rerank_strategy, top_k_for_gen=5, skip_retrieval: bool = True):
    """
    通用 RAG 评估流程
    :param test_questions: 测试集列表
    :param rerank_strategy: 排序策略函数，接收 (question, ids, chunks) 返回 (ranked_ids, ranked_chunks)
    :param top_k_for_gen: 最终喂给生成模型的片段数量
    """
    results = []
    print(f"开始评估，策略名称: {rerank_strategy.__name__}，总计 {len(test_questions)} 个问题...")
    
    pbar = tqdm(test_questions)
    for i, qa_pair in enumerate(pbar):
        question = qa_pair["question"]
        gold_answer = qa_pair["gold_answer"]

        try:
            if skip_retrieval:
                # ---------- 直接用题目全部文本块 ----------
                # 从知识库文件中读取该题的所有块
                all_chunks = get_chunks_by_question_id(i)
                final_chunks = rerank_strategy(question, all_chunks, 5)
                ranked_ids = [] 
            else:
    
                # 1. 统一初筛 (所有算法共用同一次检索)
                initial_ids, initial_chunks, _ = retrieve(question, top_k=20)
                
                # 2. 执行具体的排序策略
                # 这个函数会在内部调用你写的各类算法
                ranked_ids, ranked_chunks = rerank_strategy(question, initial_ids, initial_chunks)
                
                # 3. 截取最强的 N 个片段用于生成
                final_chunks = ranked_chunks[:top_k_for_gen]
                    
            # 4. DeepSeek 生成答案
            generated_answer = generate(question, final_chunks)
                
            # 5. 记录信息
            item = {
                "id": i,
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "retrieved_chunks": final_chunks,
                "ids": ", ".join(ranked_ids) # 记录完整的排序列表，方便算 MRR
            }
            results.append(item)
                
        except Exception as e:
            print(f"问题 {i} 出错: {e}")
            pbar.set_postfix(error=str(e)[:20])
            continue
            
    return results

# --- 策略 1: 基线 (不重排) ---
def strategy_baseline(q, ids, chunks):
    return ids, chunks

# --- 策略 2: BGE Pointwise ---
def strategy_pointwise(q, ids, chunks):
    # 这里调用你之前的 rerank_chunks
    r_ids, r_chunks, _ = bge_pointwise_rerank(q, ids, chunks)
    return r_ids, r_chunks

# --- 策略 3: ColBERT ---
def strategy_colbert(q, ids, chunks):
    return colbert_rerank(q, ids, chunks)

# --- 策略 4: DeepSeek Listwise ---
def strategy_listwise(q, ids, chunks):
    return deepseek_listwise_rerank(q, ids, chunks)

# --- 策略 5: DeepSeek Pairwise ---
def strategy_pairwise(q, ids, chunks):
    return deepseek_pairwise_rerank(q, ids, chunks)

def strategy_setselection(q, chunks, top_k):
    return deepseek_setselection(q, chunks, top_k)

def strategy_colbert_listwise (q, ids, chunks):
    c_ids, c_chunks = colbert_rerank(q, ids, chunks)
    # 2. 准备第二层细筛的数据：取 ColBERT 排序后的 Top-10
    colbert_top_10_ids = c_ids[:10]
    colbert_top_10_chunks = c_chunks[:10]
    # 3. 第二层细筛：使用 DeepSeek 对这 10 个进行 Listwise 重排
    # 它返回的是这 10 个片段的新顺序
    d_ids, d_chunks = deepseek_listwise_rerank(q, colbert_top_10_ids, colbert_top_10_chunks)
    # 4. 获取 ColBERT 漏斗中剩下的后 10 个片段 (第 11 到 20 名)
    remaining_ids = c_ids[10:]
    remaining_chunks = c_chunks[10:]
    # 5. 【核心合并步骤】
    # 将 DeepSeek 重排后的前 10 名，与 ColBERT 认为的后 10 名拼接
    final_ids = list(d_ids) + list(remaining_ids)
    final_chunks = list(d_chunks) + list(remaining_chunks)
    # 返回完整的 20 个按新优先级排列的列表
    return final_ids, final_chunks

def calculate_metrics(gt_file_path, pred_file_path):
    # 1. 读取标准答案文件
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)['test']
    
    # 2. 读取预测结果文件
    with open(pred_file_path, 'r', encoding='utf-8') as f:
        pred_list = json.load(f)

    # 将预测结果转化为以 ID 为键的字典，方便查找
    # 核心处理：将 "ids": "题_0_块_2, 题_0_块_4..." 转化为列表
    preds_by_id = {}
    for item in pred_list:
        q_id = item['id']
        if 'ids' in item and isinstance(item['ids'], str):
            # 分割字符串并去除多余空格
            retrieved_ids = [idx.strip() for idx in item['ids'].split(',')]
            preds_by_id[q_id] = retrieved_ids
        else:
            preds_by_id[q_id] = []

    # 3. 开始计算指标
    hit_at_1_scores = []
    hit_at_2_scores = []
    hit_at_20_scores = []
    mrr_scores = []
    
    # 统计成功匹配的数量
    evaluated_count = 0

    for gt_item in gt_data:
        q_id = gt_item['id']
        gt_chunks = gt_item['ground_truth_chunks'] # 这是一个列表，如 ["题_0_块_2"]
        
        if q_id not in preds_by_id:
            print(f"跳过问题 ID {q_id}: 在结果文件中未找到")
            continue
        
        evaluated_count += 1
        retrieved_ids = preds_by_id[q_id]

        # --- Hit Rate @ 1 ---
        # 检索出的第 1 个是否在正确答案列表中
        if retrieved_ids and retrieved_ids[0] in gt_chunks:
            hit_at_1_scores.append(1)
        else:
            hit_at_1_scores.append(0)

        # --- Hit Rate @ 2 (新增：你的生成模型输入上限) ---
        # 检查前 2 个 ID 中是否有任意一个在正确答案中
        top_2_retrieved = retrieved_ids[:2]
        if any(r_id in gt_chunks for r_id in top_2_retrieved):
            hit_at_2_scores.append(1)
        else:
            hit_at_2_scores.append(0)
            
        # --- Hit Rate @ 20 ---
        # 检查检索出的前 20 个 ID 中是否有任意一个在正确答案列表中
        # 使用切片 [:20] 确保只检查前20个
        top_20_retrieved = retrieved_ids[:20] 
        
        # any() 函数会检查列表中是否至少有一个匹配
        if any(r_id in gt_chunks for r_id in top_20_retrieved):
            hit_at_20_scores.append(1)
        else:
            hit_at_20_scores.append(0)
        # --- MRR (Mean Reciprocal Rank) ---
        # 找到正确答案在检索列表中的第一个位置
        rr = 0
        for rank, r_id in enumerate(retrieved_ids):
            if r_id in gt_chunks:
                rr = 1 / (rank + 1)
                break
        mrr_scores.append(rr)
        
    # 4. 打印最终结果
    if evaluated_count == 0:
        print("错误：没有匹配到任何可评估的数据。")
        return
    
    print("\n" + "="*40)
    print(f"检索效果评估报告")
    print("-" * 40)
    print(f"评估题目总数:  {evaluated_count}")
    print(f"Hit Rate @ 1:  {np.mean(hit_at_1_scores):.4f}")
    print(f"Hit Rate @ 2:  {np.mean(hit_at_2_scores):.4f}")
    print(f"Hit Rate @ 20: {np.mean(hit_at_20_scores):.4f}")
    print(f"MRR:           {np.mean(mrr_scores):.4f}")
    print("="*40)

def normalize_zh(s):
    """
    中文标准化：转换为小写，移除空格和中英文标点符号
    """
    if not s:
        return ""
    s = s.lower()
    # 移除英文标点
    s = s.translate(str.maketrans('', '', string.punctuation))
    # 移除中文标点
    s = re.sub(r'[！？。，（）：；“”‘’、—…《》]', '', s)
    # 移除所有空白字符
    s = re.sub(r'\s+', '', s)
    return s

def calculate_item_metrics(prediction, ground_truth):
    """
    计算单条结果的 P, R, F1 和 EM
    """
    pred_norm = normalize_zh(prediction)
    gt_norm = normalize_zh(ground_truth)
    
    # 1. 计算 EM (Exact Match)
    em = 1 if pred_norm == gt_norm else 0
    
    # 如果预测或金标准为空
    if not pred_norm or not gt_norm:
        # 如果都为空，F1为1；否则为0
        f1 = 1.0 if pred_norm == gt_norm else 0.0
        return 0.0, 0.0, f1, em

    # 2. 计算 P, R, F1 (字符级别)
    pred_chars = list(pred_norm)
    gt_chars = list(gt_norm)
    
    common = Counter(pred_chars) & Counter(gt_chars)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0, 0.0, 0.0, em
    
    precision = 1.0 * num_same / len(pred_chars)
    recall = 1.0 * num_same / len(gt_chars)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1, em

def run_gen_evaluation(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    p_list, r_list, f1_list, em_list = [], [], [], []

    for item in data:
        gen = item.get("generated_answer", "")
        gold = item.get("gold_answer", "")
        
        p, r, f1, em = calculate_item_metrics(gen, gold)
        
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
        em_list.append(em)

    print("\n" + "="*45)
    print(f"生成质量评估报告: {file_path}")
    print("-" * 45)
    print(f"评估样本数: {len(data)}")
    print(f"Exact Match (EM): {np.mean(em_list):.4f}")
    print(f"F1 Score:         {np.mean(f1_list):.4f}")
    print(f"Precision (P):    {np.mean(p_list):.4f}")
    print(f"Recall (R):       {np.mean(r_list):.4f}")
    print("="*45)

