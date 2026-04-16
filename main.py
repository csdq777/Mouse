import ssl
import certifi
import asyncio
import aiohttp
import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm


# =====================================================
# 基础配置
# =====================================================

ssl_context = ssl.create_default_context(cafile=certifi.where())

API_KEY = "" 
BASE_URL = ""
MODEL = "mistral-large-latest"
PROMPT_VERSION = "mistral-large-latest"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

SAVE_EVERY_N = 10
MAX_RETRY = 2
TIMEOUT = 60


# =====================================================
# 底层 API 调用
# =====================================================

async def call_api(
    session: aiohttp.ClientSession,
    prompt: str,
    retry: int = MAX_RETRY,
    timeout: int = TIMEOUT,
) -> Optional[str]:

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        # "thinking": {"type": "disabled"},
        # "reasoning"={"effort": "none"},
        # "reasoning_effort":"minimal",
        # "reasoning_effort": "minimal",
        "temperature": 0,
        "max_tokens": 128,
    }

    for attempt in range(retry + 1):
        try:
            async with session.post(
                f"{BASE_URL}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:

                if resp.status != 200:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                    continue

                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()

        except Exception:
            await asyncio.sleep(2 ** attempt)

    return None


# =====================================================
# Prompt 
# =====================================================

def build_translate_prompt(text: str) -> str:
    return f"""你是一位“抽象话”翻译专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：将提供的抽象话准确翻译成标准汉语。

要求：只输出翻译后的汉语句子，不要添加任何解释或额外内容。

抽象话：{text}

汉语："""


def build_toxicity_prompt(text: str) -> str:
    return f"""你是一位“抽象话”内容识别安全专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：判断给定的文本是否具有“有毒性”。

请按照以下步骤进行内部判断：
1. 去混淆与解码：识别并还原文本中的拼音首字母（如NMSL）、谐音字（如沙雕、草泥马）、拆字（如杀.人）或Emoji指代（如🐴）。
2. 意图分析：判断还原后的文本在当前语境下的真实意图。
3. 分类判定：
   - 如果文本包含辱骂、人身攻击、仇恨言论、暴力威胁、性骚扰或极度恶意的讽刺，视为“有毒”。
   - 如果文本是正常交流、陈述事实、非恶意的玩笑、自嘲（例如“笑死”、“我太菜了”）或积极内容，视为“无毒”。

要求：仅输出一个数字，0代表无毒，1代表有毒，严禁输出任何解释、标点符号或其他文字。

抽象话：{text}

毒性："""


def build_intention_prompt(text: str) -> str:
    return f"""你是一位“抽象话”意图识别专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：请根据文本的隐含语义与语境，判断提供的抽象话属于什么意图。选项有以下：
0 评价（如吐槽、夸赞等）
1 一般陈述
2 群体认同
3 幽默和玩梗
4 情绪表达
5 性化指代（将涉及性或色情的词汇归于特定的主体）
6 指令催促
7 其他

要求：只输出序号即可，不要添加任何解释或额外内容。

抽象话：{text}

意图分类："""


def build_homephone_prompt(text: str) -> str:
    return f"""你是一位擅长分析“抽象话”中是否含有谐音替换的专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：判断给定的文本是否有谐音替换。

请按照以下步骤进行内部判断：
1. 给定句子可能有以下成分：汉字、Emoji、符号、数字、拼音、方言、公式等方式进行的谐音表达（包括同音或近音替换，且不限于汉语或其他语言）。

要求：仅输出一个数字，0代表无，1代表有，严禁输出任何解释、标点符号或其他文字。

抽象话：{text}

分类："""


def build_semantic_prompt(text: str) -> str:
    return f"""你是一位擅长分析“抽象话”中是否含有语义转换成分的专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：判断给定的文本是否有语义转换。

请按照以下步骤进行内部判断：
1. 给定句子可能有以下成分：社区特定称谓、成分同义替换（包括Emoji/文字/符号等形式）、网络梗表达、一门语言在语音角度倒放，以及对特定事物的指代。

要求：仅输出一个数字，0代表无，1代表有，严禁输出任何解释、标点符号或其他文字。

抽象话：{text}

分类："""


def build_vision_prompt(text: str) -> str:
    return f"""你是一位擅长分析“抽象话”中是否含有视觉类推成分的专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：判断给定的文本是否有视觉类推。

请按照以下步骤进行内部判断：
1. 给定句子可能有以下成分：偏旁拆分（如“亻尔”表示“你”）、字形替换（如“搞劳”代替“犒劳”）、Emoji的视觉转义（如🍆表“紫色”）、数字/符号的视觉象征（如3表“亲亲”、Ψ表可能不是“叉子”而是“三”）等，需结合上下文语义进行视觉类推。

要求：仅输出一个数字，0代表无，1代表有，严禁输出任何解释、标点符号或其他文字。

抽象话：{text}

分类："""


def build_cloze_prompt(text: str, options: str) -> str:
    return f"""你是一位“抽象话”上下文完形填空专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：提供给你一个中文互联网上的抽象话对话场景及其对应选项，选择一个最合适的选项使得对话合理完整。

要求：只输出选项字母即可，不要添加任何解释或额外内容。

题目：
{text}

选项：
{options}

结果："""


def build_choice_prompt(text: str, a: str, b: str, c: str) -> str:
    return f"""你是一位“抽象话”单选题匹配专家。所谓抽象话，是由谐音、视觉、语义三大类手法组合而成的特殊表达方式，具体包括谐音替换（汉字、Emoji、符号、数字、拼音、方言、公式等）、视觉类推（偏旁、Emoji、汉字、符号、数字）以及语义转换（社区指代、同义替换、网络梗、语言倒放、实体指代等）。

任务：提供给你抽象话相关的题目，请从3个选项中选择一项和题目含义最匹配的选项。

要求：只输出选项字母即可，不要添加任何解释或额外内容。

题目：{text}

选项：
{a}
{b}
{c}

结果："""


# =====================================================
# 单条样本
# =====================================================

async def eval_one_text(session, sem, text: str, row):
    async with sem:
        task_defs = {
            "llm_translation": ("prompt_translate", build_translate_prompt),
            "llm_toxicity": ("prompt_toxicity", build_toxicity_prompt),
            "llm_intention": ("prompt_intention", build_intention_prompt),
            "llm_homephone": ("prompt_homephone", build_homephone_prompt),
            "llm_semantic": ("prompt_semantic", build_semantic_prompt),
            "llm_vision": ("prompt_vision", build_vision_prompt),
        }

        prompts = {}
        tasks = {}
        
        for col, (prompt_key, builder) in task_defs.items():
            if pd.isna(row[col]):
                prompt = builder(text)
                prompts[prompt_key] = prompt
                tasks[col] = call_api(session, prompt)

        if not tasks:
            return {}, {}  # 这一行啥都不需要生成

        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results)), prompts



# =====================================================
# 主 CSV 评测
# =====================================================


async def eval_main_csv(input_csv, output_csv, max_concurrency=3):
    df = pd.read_csv(input_csv)

    result_cols = [
        "llm_translation",
        "llm_toxicity",
        "llm_intention",
        "llm_homephone",
        "llm_semantic",
        "llm_vision",
        "prompt_version",
        "error",
    ]

    prompt_cols = [
        "prompt_translate",
        "prompt_toxicity",
        "prompt_intention",
        "prompt_homephone",
        "prompt_semantic",
        "prompt_vision",
    ]

    for c in result_cols + prompt_cols:
        if c not in df.columns:
            df[c] = None

    df.to_csv(output_csv, index=False)

    sem = asyncio.Semaphore(max_concurrency)

    async with aiohttp.ClientSession(
        headers=HEADERS,
        connector=aiohttp.TCPConnector(ssl=ssl_context),
    ) as session:


        need_cols = [
            "llm_translation",
            "llm_toxicity",
            "llm_intention",
            "llm_homephone",
            "llm_semantic",
            "llm_vision",
        ]

        rows = df[df[need_cols].isna().any(axis=1)]

        tasks = [
            (
                idx,
                asyncio.create_task(
                    eval_one_text(
                        session,
                        sem,
                        row["Source language"],
                        row,
                    )
                ),
            )
            for idx, row in rows.iterrows()
        ]



        for i, (idx, task) in enumerate(tqdm(tasks)):
            try:
                (results, prompts) = await task
                for col, val in results.items():
                    df.at[idx, col] = val

                for k, v in prompts.items():
                    df.at[idx, k] = v

                df.at[idx, "prompt_version"] = PROMPT_VERSION
                df.at[idx, "error"] = 0


            except Exception:
                df.at[idx, "error"] = 1

            if i % SAVE_EVERY_N == 0:
                df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)

async def eval_cloze_csv(input_csv, output_csv, max_concurrency=3):
    df = pd.read_csv(input_csv)

    result_cols = ["llm_choice", "prompt_cloze", "prompt_version", "error"]
    for c in result_cols:
        if c not in df.columns:
            df[c] = None

    sem = asyncio.Semaphore(max_concurrency)

    async def one_task(idx, row, session):
        async with sem:
            prompt = build_cloze_prompt(row["题目"], row["选项"])
            try:
                res = await call_api(session, prompt)
                return idx, res, prompt, 0
            except Exception:
                return idx, None, prompt, 1

    async with aiohttp.ClientSession(
        headers=HEADERS,
        connector=aiohttp.TCPConnector(ssl=ssl_context),
    ) as session:

        tasks = [
            one_task(idx, row, session)
            for idx, row in df.iterrows()
            if pd.isna(row["llm_choice"])
        ]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Cloze Eval"):
            idx, choice, prompt, err = await coro
            df.at[idx, "llm_choice"] = choice
            df.at[idx, "prompt_cloze"] = prompt
            df.at[idx, "prompt_version"] = PROMPT_VERSION
            df.at[idx, "error"] = err

    df.to_csv(output_csv, index=False)


async def eval_choice_csv(input_csv, output_csv, max_concurrency=3):
    df = pd.read_csv(input_csv)

    result_cols = ["llm_choice", "prompt_choice", "prompt_version", "error"]
    for c in result_cols:
        if c not in df.columns:
            df[c] = None

    sem = asyncio.Semaphore(max_concurrency)

    async def one_task(idx, row, session):
        async with sem:
            prompt = build_choice_prompt(
                row["题目"],
                row["A选项"],
                row["B选项"],
                row["C选项"],
            )
            try:
                res = await call_api(session, prompt)
                return idx, res, prompt, 0
            except Exception:
                return idx, None, prompt, 1

    async with aiohttp.ClientSession(
        headers=HEADERS,
        connector=aiohttp.TCPConnector(ssl=ssl_context),
    ) as session:

        tasks = [
            one_task(idx, row, session)
            for idx, row in df.iterrows()
            if pd.isna(row["llm_choice"])
        ]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Choice Eval"):
            idx, choice, prompt, err = await coro
            df.at[idx, "llm_choice"] = choice
            df.at[idx, "prompt_choice"] = prompt
            df.at[idx, "prompt_version"] = PROMPT_VERSION
            df.at[idx, "error"] = err

    df.to_csv(output_csv, index=False)



if __name__ == "__main__":


## For concurrency, please follow the service rules

    asyncio.run(
        eval_main_csv(
            input_csv="./data/main_eval.csv",
            output_csv="",
            max_concurrency=1,
        )
    )

    asyncio.run(
        eval_cloze_csv(
            input_csv="./data/icc.csv",
            output_csv="",
            max_concurrency=1,
        )
    )

    asyncio.run(
        eval_choice_csv(
            input_csv="./data/single_choice.csv",
            output_csv="",
            max_concurrency=1, 
        )
    )
