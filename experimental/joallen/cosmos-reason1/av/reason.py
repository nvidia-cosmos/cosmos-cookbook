import argparse
import concurrent.futures
import glob
import json
import os
import time

import openai
from openai.types.chat import ChatCompletionUserMessageParam
from tqdm import tqdm


"""
Get the reasoning trace and next driving behavior prediction by ds-r1
"""


# initialize ds-r1 client
client = openai.OpenAI(
    base_url="http://10.12.189.186:8000/v1",
    api_key="input-your-api-key",
)


# predict next driving behavior and get related reasoning trace
def predict(caption: str) -> tuple:
    prompt = f"""
Below is a driving scenario:

{caption}

Please think through and predict the next ego driving behavior in the following format:
<action> your predicted action </action>.
""".strip()

    messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]
    response = client.chat.completions.create(model="deepseek-reasoner", messages=messages, temperature=0.6)

    content = response.choices[0].message.content
    reasoning = content.split("<think>")[1].split("</think>")[0].strip()
    content = content.split("</think>")[1].strip()
    prediction = content.split("<action>")[1].split("</action>")[0].strip()

    result = {"reasoning": reasoning, "prediction": prediction}

    return prompt, result


# process a single clip
def process_single_clip(caption_file: str) -> tuple:
    with open(caption_file, "r") as f:
        data = json.load(f)

    try:
        time_s = time.time()
        prompt, result = predict(data["response"])
        time_e = time.time()
        return time_e - time_s, prompt, result
    except Exception as e:  # noqa: BLE001
        print(f"error processing {caption_file}: {e}")
        return 0, "", None


# build the data to be saved
def build_data(prompt: str, time_used: float, result: dict) -> dict:
    return {
        "user_prompt_during_inference": (
            "You are given a driving video. Please carefully analyze the video content, then think through and predict "
            "the next ego driving behavior in the following format: <action> your predicted action </action>."
        ),
        "system_prompt": "",
        "user_prompt": prompt,
        "prompt_version": "v1",
        "timestamp": time_used,
        "response": result,
        "success": "True" if result is not None else "False",
    }


# process all clips by a single thread (for debugging only)
def process_clips_single_thread(caption_files: list, dir_reasoning: str) -> None:
    for caption_file in caption_files:
        time_used, prompt, result = process_single_clip(caption_file)
        data = build_data(prompt, time_used, result)

        reasoning_file = os.path.join(dir_reasoning, os.path.basename(caption_file))
        with open(reasoning_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"clip done: {caption_file}")


# process all clips by multiple threads
def process_clips_multi_thread(caption_files: list, dir_reasoning: str, workers: int = 64) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for caption_file in caption_files:
            futures.append(executor.submit(process_single_clip, caption_file))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="processing clips"):
            time_used, prompt, result = future.result()
            data = build_data(prompt, time_used, result)

            reasoning_file = os.path.join(dir_reasoning, os.path.basename(caption_file))
            with open(reasoning_file, "w") as f:
                json.dump(data, f, indent=2)


def main(args: argparse.Namespace) -> None:
    dir_captions = args.dir_captions
    dir_reasoning = args.dir_reasoning
    caption_files = glob.glob(os.path.join(dir_captions, "*.json"))

    process_clips_multi_thread(caption_files, dir_reasoning)

    # for debugging purposes, use single-thread processing
    # process_clips_single_thread(caption_files, dir_reasoning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_captions", type=str, default="/home/xiaodongy/research/data/av_reasoning/captions")
    parser.add_argument("--dir_reasoning", type=str, default="/home/xiaodongy/research/data/av_reasoning/reasoning")
    args = parser.parse_args()
    main(args)
