from pathlib import Path
import json
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
LOCAL_FILE = "./multifieldqa_zh.jsonl"

def load_longbench_with_refined_chunking():
    if not os.path.exists(LOCAL_FILE):
        print(f"❌ 错误：找不到文件 {LOCAL_FILE}。")
        return
    
    print(f"🚀 正在启动：智能清洗 + 递归分块模式...")
    
    # 初始化递归分块器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 针对 BGE-M3，建议 800 左右，信息熵最均衡
        chunk_overlap=150, # 增加重叠度，防止多跳推理的线索被切断
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        length_function=len
    )

    knowledge_base = []
    test_questions = []

    with open(LOCAL_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 50: break 
            
            item = json.loads(line)
            question = item['input']
            full_text = item['context']
            
            # --- [新增] 深度清洗：在分块前剔除杂质 ---
            # 1. 移除目录引导线 (点号、短横线、下划线)
            full_text = re.sub(r'[·\. \-_]{5,}', ' ', full_text)
            # 2. 移除由于爬取或 PDF 转换产生的非法 Unicode 符号
            full_text = re.sub(r'[\uE000-\uF8FF]', '', full_text)
            # 3. 将多余的空白符压缩为单个空格
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # 执行智能分块
            chunks = text_splitter.split_text(full_text)
            
            # 入库并打标
            for j, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) > 15:
                    # 格式：[题_i_块_j] 内容
                    indexed_chunk = f"[题_{i}_块_{j}] {chunk_text.strip()}"
                    knowledge_base.append(indexed_chunk)
            
            # 答案处理
            gold_answer = item['answers'][0] if isinstance(item['answers'], list) else item['answers']
            
            test_questions.append({
                "id": i,
                "question": question,
                "gold_answer": gold_answer,
                "chunk_count": len(chunks)
            })

    # 保存文件 (逻辑保持不变)
    with open("longbench_kb.md", "w", encoding="utf-8") as f:
        for chunk in knowledge_base:
            f.write(chunk + "\n\n---\n\n")

    with open("longbench_eval.json", "w", encoding="utf-8") as f:
        json.dump({"test": test_questions}, f, indent=2, ensure_ascii=False)

    print(f"✅ 构建完成！知识库块数: {len(knowledge_base)}，测试题数: {len(test_questions)}")
    print(f"📊 样例：第一题被切分为 {test_questions[0]['chunk_count']} 个语义连贯的知识块。")



def split_chunks_with_cleaning(file_path: str) -> List[str]:
    """
    读取文件，执行深度清洗，并根据 '---' 分隔符分块
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # 1. 深度清洗逻辑
    # a. 将 <br> 或 <br/> 替换为换行符
    content = re.sub(r'<br\s*/?>', '\n', content)
    
    # b. 剔除 <img> 标签及其内容
    content = re.sub(r'<img>.*?</img>', '', content)
    content = re.sub(r'<img>', '', content)
    
    # c. 剔除常见的 HTML 转义字符（如 &nbsp;, &gt; 等）
    content = re.sub(r'&[a-z]+;', ' ', content)
    
    # d. 剔除特殊的控制字符或乱码前缀（如 \ue50a 等）
    content = re.sub(r'[\uE000-\uF8FF]', '', content)

    # 2. 按分隔符切分
    raw_blocks = content.split('\n\n---\n\n')
    
    final_chunks = []
    for block in raw_blocks:
        # 3. 针对每个块的精细化清理
        # 替换多个连续空格为一个空格
        text = re.sub(r'\s+', ' ', block)
        text = text.strip()
        
        # 过滤掉过短的块
        if len(text) > 15:
            final_chunks.append(text)
                
    return final_chunks

def update_eval_file(input_file, output_file):
    # 手工分析的映射数据 (question_id -> ground_truth_chunk_id)
    gt_mapping = {
        0: "题_0_块_2", 1: "题_1_块_5", 2: "题_2_块_3", 3: "题_3_块_5", 4: "题_4_块_1",
        5: "题_5_块_0", 6: "题_6_块_1", 7: "题_7_块_0", 8: "题_8_块_0", 9: "题_9_块_0",
        10: "题_10_块_0", 11: "题_11_块_0", 12: "题_12_块_0", 13: "题_13_块_1", 14: "题_14_块_0",
        15: "题_15_块_0", 16: "题_16_块_3", 17: "题_17_块_1", 18: "题_18_块_8", 19: "题_19_块_0",
        20: "题_20_块_4", 21: "题_21_块_2", 22: "题_22_块_1", 23: "题_23_块_2", 24: "题_24_块_9",
        25: "题_25_块_8", 26: "题_26_块_0", 27: "题_27_块_2", 28: "题_28_块_0", 29: "题_29_块_0",
        30: "题_30_块_1", 31: "题_31_块_10", 32: "题_32_块_5", 33: "题_33_块_0", 34: "题_34_块_4",
        35: "题_35_块_4", 36: "题_36_块_6", 37: "题_37_块_3", 38: "题_38_块_8", 39: "题_39_块_2",
        40: "题_40_块_1", 41: "题_41_块_2", 42: "题_42_块_5", 43: "题_43_块_15", 44: "题_44_块_12",
        45: "题_45_块_3", 46: "题_46_块_0", 47: "题_47_块_3", 48: "题_48_块_0", 49: "题_49_块_2"
    }

    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 更新数据
    for item in data['test']:
        q_id = item['id']
        if q_id in gt_mapping:
            # 使用列表存储，方便后续支持一题多块的情况
            item['ground_truth_chunks'] = [gt_mapping[q_id]]
        else:
            item['ground_truth_chunks'] = []

    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"成功！已生成带标注的文件: {output_file}")



def generate_contextual_chunks(kb_path):
    # 读取 KB
    with open(kb_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按 --- 分割并解析 ID 和内容
    raw_parts = content.split('---')
    docs_group = {} # 按 "题_X" 分组
    
    for part in raw_parts:
        part = part.strip()
        if not part: continue
        match = re.search(r'\[(题_(\d+)_块_(\d+))\]', part)
        if match:
            full_id = match.group(1)
            doc_id = match.group(2) # 提取题号作为文档标识
            chunk_num = match.group(3)
            text = re.sub(r'\[题_\d+_块_\d+\]', '', part).strip()
            
            if doc_id not in docs_group:
                docs_group[doc_id] = []
            docs_group[doc_id].append({"id": full_id, "text": text, "num": int(chunk_num)})

    enhanced_data = []

    # 遍历每个文档进行背景增强
    for doc_id, chunks in tqdm(docs_group.items(), desc="正在进行 Contextual 增强"):
        # 假设块 0 通常包含文档标题和背景
        doc_header = next((c['text'] for c in chunks if c['num'] == 0), chunks[0]['text'])
        
        # 批量处理：每 2-3 个块调用一次 DeepSeek
        batch_size = 2
        for i in range(0, len(chunks), batch_size):
            current_batch = chunks[i : i + batch_size]
            
            # 构造 Prompt
            chunks_to_analyze = ""
            for idx, item in enumerate(current_batch):
                chunks_to_analyze += f"<chunk_{idx}>\n{item['text']}\n</chunk_{idx}>\n\n"

            prompt = f"""你是一个文档分析专家。请为下面这几个文本块分别编写一段极简的上下文背景（Context）。
背景信息：
{doc_header}

待处理的文本块：
{chunks_to_analyze}

任务要求：
1. 为每个 chunk 编写 1 句话的背景（包含：文档主题、当前段落属于什么章节/范畴）。
2. 背景必须能够辅助检索，明确该段落属于哪一年、哪个城市或哪个具体案例。
3. 请以 JSON 列表格式输出，键名为 "contexts"，包含与块数量一致的字符串。

示例输出格式：
{{ "contexts": ["这段话摘自2012年铜仁市报告，涉及工业经济增长...", "这段话关于..."] }}
"""
            try:
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={'type': 'json_object'}, # 强制输出 JSON
                    temperature=0.1
                )
                res_data = json.loads(response.choices[0].message.content)
                contexts = res_data.get("contexts", [""] * len(current_batch))
                
                # 融合 Context 和 Original Text
                for item, ctx in zip(current_batch, contexts):
                    # 将背景放在正文前面，用换行符隔开
                    combined_text = f"[Context: {ctx}]{item['text']}"
                    enhanced_data.append({"id": item['id'], "text": combined_text})
            except Exception as e:
                print(f"DeepSeek 增强失败，使用原始文本：{e}")
                for item in current_batch:
                    enhanced_data.append({"id": item['id'], "text": item['text']})

    return enhanced_data

def load_longbench_dataset_with_chunking():
    # 1. 配置路径
    dataset_path = str(Path(__file__).resolve().parents[1] / "data/longbench_dureader")  # 你保存数据集的目录
    
    if not os.path.exists(dataset_path):
        print(f"❌ 错误：找不到数据集目录 {dataset_path}。")
        return
    
    print(f"🚀 正在启动：从 LongBench dureader 数据集读取 -> 智能清洗 + 递归分块模式...")
    
    # 2. 加载数据集（这一行取代了之前逐行读 JSONL）
    dataset = load_from_disk(dataset_path)
    test_data = dataset['test']  # 取出 test 切片的 200 条数据

    # 3. 初始化递归分块器（完全保留你的参数）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        length_function=len
    )

    knowledge_base = []
    test_questions = []

    # 4. 遍历数据集（取代了之前的 enumerate(line)）
    for i, sample in enumerate(test_data):
        if i >= 50:  # 保留你的测试限制
            break
        
        question = sample['input']
        full_text = sample['context']
        
        # --- [保留] 深度清洗逻辑 ---
        # 1. 移除目录引导线
        full_text = re.sub(r'[·\. \-_]{5,}', ' ', full_text)
        # 2. 移除非法 Unicode 符号
        full_text = re.sub(r'[\uE000-\uF8FF]', '', full_text)
        # 3. 压缩多余空白符
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # 执行智能分块
        chunks = text_splitter.split_text(full_text)
        
        # 入库并打标（格式保持不变）
        for j, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) > 15:
                indexed_chunk = f"[题_{i}_块_{j}] {chunk_text.strip()}"
                knowledge_base.append(indexed_chunk)
        
        # 答案处理（逻辑保留）
        gold_answer = sample['answers'][0] if isinstance(sample['answers'], list) else sample['answers']
        
        test_questions.append({
            "id": i,
            "question": question,
            "gold_answer": gold_answer,
            "chunk_count": len(chunks)
        })

    # 5. 保存文件（完全保留）
    with open("dureader_kb.md", "w", encoding="utf-8") as f:
        for chunk in knowledge_base:
            f.write(chunk + "\n\n---\n\n")

    with open("dureader_eval.json", "w", encoding="utf-8") as f:
        json.dump({"test": test_questions}, f, indent=2, ensure_ascii=False)

    print(f"✅ 构建完成！知识库块数: {len(knowledge_base)}，测试题数: {len(test_questions)}")
    if test_questions:
        print(f"📊 样例：第一题被切分为 {test_questions[0]['chunk_count']} 个语义连贯的知识块。")