import os
import asyncio
import pandas as pd
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
import difflib



API_KEY = "" 
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"



INPUT_FILE_PATH='data\Mistral_large_3.xlsx'


SHEET_NAME = 'data'

TEXT_NAME = 'Source language'
REFERENCE_NAME = "Reference"
LLM_TRANSLATION_NAME = "llm_translation"
COLUMN_NAME="deepseek_similarity"


MAX_WORKERS = 10  
REQUEST_TIMEOUT = 30  


aclient = None

def init_client():
    
    global aclient
    if aclient is None:
        aclient = AsyncOpenAI(
            api_key=API_KEY or os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=BASE_URL,
            timeout=REQUEST_TIMEOUT,
        )


async def call_api(content):
    
    try:
        response = await aclient.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":"You are a helpful assistant."},
                {"role": "user", "content": content}
            ],
            temperature=0,
            # extra_body={"thinking": {"type": "enabled"}},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用错误: {e}")
        return None
    


async def similarity_evaluation(text, reference, llm_translation):
    
    diff_hint = get_sentence_diff_hint(text, reference)
    translation_score_prompt = f"""输入信息

源文本：{text}

参考答案：{reference}

得分点/关键词：{diff_hint}

考生作答：{llm_translation}

评估步骤（请严格按顺序执行）

第一步（反作弊检查 - 关键）：
检查“考生作答”是否与“源文本”完全相同或高度相似（仅照抄原文）。

如果考生只是照抄了源文本而未进行翻译 -> 直接判为 0 分，停止后续判断。

第二步（判断满分）：
比较“考生作答”与“参考答案”。如果两者的核心语义一致（允许表达方式的合理多样性，不要求逐字相同），且符合源文本的原意 -> 得 2 分，停止后续判断。

第三步（判断得分点）：
如果第一步通过但未达到2分标准（语义有偏差或不通顺），请检查“考生作答”中是否包含“得分点”中的关键信息。

如果包含得分点内容 -> 得 1 分。

如果不包含得分点内容 -> 得 0 分。

第四步（兜底）：
如果以上都不满足 -> 得 0 分。

输出格式

请先进行简短分析（50字以内），特别是如果判0分，请说明是因为照抄原文还是语义错误。最后一行严格输出分数。
格式如下：
分析：[简短分析理由]
分数：x"""
    llm_compare_score = await call_api(translation_score_prompt)

    print(f"Reference:[{reference}], LLM translation: [{llm_translation}],Similarity output:[{llm_compare_score}]")
    print(f"{diff_hint}")
    score = re.search(r"分数：[0-2]", llm_compare_score).group(0)
    print(f"Score: [{score}]")
    return score



def get_sentence_diff_hint(source_sentence: str, target_sentence: str) -> str:
    source_words = list(source_sentence)
    target_words = list(target_sentence)

    matcher = difflib.SequenceMatcher(None, source_words, target_words)

    diff_parts = []

    for opcode, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        source_segment = "".join(source_words[a_start:a_end])
        target_segment = "".join(target_words[b_start:b_end])

        # 提取关键差异：替换(replace)、删除(delete)、插入(insert)
        if opcode == 'replace':
            diff_parts.append(f"{{'{source_segment}'}} -> {{'{target_segment}'}} (替换)")
        elif opcode == 'delete':
            diff_parts.append(f"{{'{source_segment}'}} (被删除)")
        elif opcode == 'insert':
            diff_parts.append(f"(插入) -> {{'{target_segment}'}}")

    if not diff_parts:
        return "得分点：句子完全一致，无差异点。"

    return "得分点：" + "，".join(diff_parts)

async def translate(text):
    """翻译的提示"""
    
    translate_prompt = f"""你是一位“抽象话”翻译专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：将提供的抽象话准确翻译成标准汉语。

要求：只输出翻译后的汉语句子，不要添加任何解释或额外内容。

抽象话：{text}

汉语：
        """
    llm_translation = await call_api(translate_prompt)
    print(f"Text :[{text}], Translation of LLM :[{llm_translation}]")
    return llm_translation




async def evaluate_toxicity_for_row(row_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    
    try:
        text = row_data.get(TEXT_NAME, "")
        reference = row_data.get(REFERENCE_NAME,"")
        llm_translation = row_data.get(LLM_TRANSLATION_NAME, "")
        
        text = row_data.get(TEXT_NAME, "")

        data  = await similarity_evaluation(text, reference, llm_translation)

        return {
            "index": index,
            "data": data
        }
        
    except Exception as e:
        print(f"处理第{index}条数据时出错: {e}")
        return {
            "index": index,
            "toxicity_score": None
        }

async def evaluate_toxicity_for_row_with_semaphore(row_data: Dict[str, Any], index: int, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    
    async with semaphore:
        return await evaluate_toxicity_for_row(row_data, index)

async def process_all_rows_async(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    
    init_client()
    
    pending_indices = []
    for i in df.index:
        score = df.at[i, column_name]
        if pd.isna(score) or str(score).strip() == "":
            pending_indices.append(i)
            
    num_to_process = len(pending_indices)
    
    
    if num_to_process == 0:
        return df

   
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    
   
    tasks = []
    for i in pending_indices:
       
        row_data = df.loc[i].to_dict()
        task = evaluate_toxicity_for_row_with_semaphore(row_data, i, semaphore)
        tasks.append(task)
    
    # 4. 并发执行
    # 注意：tqdm total 设置为 len(tasks) 即待处理数量
    results = []
    save_counter = 0 
    
    for task in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理进度"):
        try:
            result = await task
            results.append(result)
            
            # 更新DataFrame
            idx = result["index"]
            df.at[idx, column_name] = result["data"]
            
            save_counter += 1
            
                
        except Exception as e:
            print(f"任务执行异常: {e}")
    
    with pd.ExcelWriter(INPUT_FILE_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=SHEET_NAME, index=False)
    print("所有任务执行完毕，结果已保存。")
    
    return df



def main():
    
    print(f"正在读取文件: {INPUT_FILE_PATH}")
    
    if not os.path.exists(INPUT_FILE_PATH):
        print("错误：文件不存在，请检查路径。")
        return

    df = pd.read_excel(INPUT_FILE_PATH, sheet_name=SHEET_NAME)
    
    df = asyncio.run(process_all_rows_async(df, COLUMN_NAME))
    
    


if __name__ == "__main__":

    
    main()