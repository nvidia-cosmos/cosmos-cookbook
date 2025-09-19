import argparse
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from pipelines.benchmarking.v1.scores.qa.prompts.qa_text2world import templates
from pipelines.cub.utils import io_utils, prompting_utils

PROMPT_ADAPTER_TEMPLATES = {
    "prompt_templates": templates,
}


def parse_datasource_args() -> argparse.Namespace:
    """Parse command line arguments for datasource pipelines."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_caption_dir",
        type=str,
        required=True,
        help="Directory containing caption JSON files",
    )
    parser.add_argument("--output_vqa_dir", type=str, required=True, help="Directory to save output VQA JSON files")
    parser.add_argument("--system_prompt_version", type=int, default=0)
    parser.add_argument("--user_prompt_version", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def generate_qa_from_captions(
    input_caption_dir: str,
    output_vqa_dir: str,
    system_prompt_version: int,
    user_prompt_version: int,
    seed: int,
) -> None:
    """Prompt GPT-4o to generate QAs from captions."""
    input_caption_path = Path(input_caption_dir)
    output_vqa_path = Path(output_vqa_dir)

    assert input_caption_path.exists(), f"Input caption directory not found: {input_caption_path}"
    output_vqa_path.mkdir(parents=True, exist_ok=True)

    # Get the prompt templates
    prompt_templates = PROMPT_ADAPTER_TEMPLATES["prompt_templates"]

    # Set up the OpenAI client
    client = prompting_utils.setup_oai_client()

    # Get the system and user prompts
    system_prompt = prompt_templates[f"system_template_v{system_prompt_version}"]
    user_prompt_fn = prompt_templates[f"user_template_fn_v{user_prompt_version}"]
    output_format = prompt_templates[f"output_format_v{user_prompt_version}"]

    # Get all caption JSON files
    caption_files = sorted([f for f in input_caption_path.glob("*.json")])

    for caption_file in tqdm(caption_files):
        # Create corresponding output file path
        output_json_path = output_vqa_path / caption_file.name

        try:
            # Read caption JSON
            with open(caption_file, "r") as f:
                caption_data = json.load(f)

            # Extract the "overall" caption from the first element in "response"
            if (
                "response" in caption_data
                and caption_data["response"]
                and isinstance(caption_data["response"], list)
                and "overall" in caption_data["response"][0]
            ):
                caption = caption_data["response"][0]["overall"].strip()
            else:
                logger.warning(f"Invalid caption format in {caption_file}")
                continue

            # Generate user prompt
            user_prompt = user_prompt_fn(caption)

            # Pass the composed messages to the OpenAI API
            messages = prompting_utils.create_messages_for_text_prompt(system_prompt, user_prompt)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                seed=seed,
                max_tokens=1000,
                response_format=output_format,  # structured output
            )

            # Parse the response from the OpenAI API
            output = json.loads(str(response.choices[0].message.content))
            if output is not None:
                q1 = {
                    "question": output["question 1 (Space: Relationship)"],
                    "answer": output["answer 1"],
                }
                q2 = {
                    "question": output["question 2 (Space: Relationship)"],
                    "answer": output["answer 2"],
                }
                q3 = {
                    "question": output["question 3 (Space: Interaction)"],
                    "answer": output["answer 3"],
                }
                q4 = {
                    "question": output["question 4 (Space: Interaction)"],
                    "answer": output["answer 4"],
                }
                q5 = {
                    "question": output["question 5 (Space: Geometry)"],
                    "answer": output["answer 5"],
                }
                q6 = {
                    "question": output["question 6 (Space: Geometry)"],
                    "answer": output["answer 6"],
                }
                q7 = {
                    "question": output["question 7 (Time: Actions)"],
                    "answer": output["answer 7"],
                }
                q8 = {
                    "question": output["question 8 (Time: Actions)"],
                    "answer": output["answer 8"],
                }
                q9 = {
                    "question": output["question 9 (Time: Order)"],
                    "answer": output["answer 9"],
                }
                q10 = {
                    "question": output["question 10 (Time: Order)"],
                    "answer": output["answer 10"],
                }
                q11 = {
                    "question": output["question 11 (Time: Camera)"],
                    "answer": output["answer 11"],
                }
                q12 = {
                    "question": output["question 12 (Time: Camera)"],
                    "answer": output["answer 12"],
                }
                q13 = {
                    "question": output["question 13 (Physics: Attributes)"],
                    "answer": output["answer 13"],
                }
                q14 = {
                    "question": output["question 14 (Physics: Attributes)"],
                    "answer": output["answer 14"],
                }
                q15 = {
                    "question": output["question 15 (Physics: States)"],
                    "answer": output["answer 15"],
                }
                q16 = {
                    "question": output["question 16 (Physics: States)"],
                    "answer": output["answer 16"],
                }
                q17 = {
                    "question": output["question 17 (Physics: Object Permanence)"],
                    "answer": output["answer 17"],
                }
                q18 = {
                    "question": output["question 18 (Physics: Object Permanence)"],
                    "answer": output["answer 18"],
                }
                success = True
            else:
                q1 = q2 = q3 = q4 = q5 = q6 = q7 = q8 = q9 = q10 = q11 = q12 = q13 = q14 = q15 = q16 = q17 = q18 = None
                success = False
            full_response = response.choices[0].message.content

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing {caption_file}: {str(e)}")
            q1 = q2 = q3 = q4 = q5 = q6 = q7 = q8 = q9 = q10 = q11 = q12 = q13 = q14 = q15 = q16 = q17 = q18 = None
            success = False
            full_response = str(e)

        # Create results dictionary
        results = {
            "Q1 (Space: Relationship)": q1,
            "Q2 (Space: Relationship)": q2,
            "Q3 (Space: Interaction)": q3,
            "Q4 (Space: Interaction)": q4,
            "Q5 (Space: Geometry)": q5,
            "Q6 (Space: Geometry)": q6,
            "Q7 (Time: Actions)": q7,
            "Q8 (Time: Actions)": q8,
            "Q9 (Time: Order)": q9,
            "Q10 (Time: Order)": q10,
            "Q11 (Time: Camera)": q11,
            "Q12 (Time: Camera)": q12,
            "Q13 (Physics: Attributes)": q13,
            "Q14 (Physics: Attributes)": q14,
            "Q15 (Physics: States)": q15,
            "Q16 (Physics: States)": q16,
            "Q17 (Physics: Object Permanence)": q17,
            "Q18 (Physics: Object Permanence)": q18,
            "Full Response": full_response,
            "Success": success,
        }

        # Write the results to a JSON file
        io_utils.write_json_file(results, str(output_json_path))
        logger.info(f"Results written to {output_json_path}")


if __name__ == "__main__":
    """
    Example usage:
    PYTHONPATH=$(pwd) python pipelines/benchmarking/v1/scores/qa/scripts/generate_qa_for_cosmospredict2_benchmark.py \
    --input_caption_dir "/mnt/andy/data/robot/cosmos_predict2_benchmark/format/caption" \
    --output_vqa_dir "/mnt/andy/data/robot/cosmos_predict2_benchmark/format/vqa"
    """
    args = parse_datasource_args()
    input_caption_dir = args.input_caption_dir
    output_vqa_dir = args.output_vqa_dir
    system_prompt_version = args.system_prompt_version
    user_prompt_version = args.user_prompt_version
    seed = args.seed

    generate_qa_from_captions(
        input_caption_dir=input_caption_dir,
        output_vqa_dir=output_vqa_dir,
        system_prompt_version=system_prompt_version,
        user_prompt_version=user_prompt_version,
        seed=seed,
    )
