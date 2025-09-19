import argparse
import glob
import json
import os

"""
Extract the core labeling information to compose as captions for DS-R1:
global_description                   => description
value_driving_difficulty_explanation => driving difficulity explanation
event                                => notice
"""


def compose_caption(annotation: dict) -> dict:
    prompt_v1 = """
    You are given a driving video.
    Please carefully analyze the video and describe the driver, environment and critical objects.

    For the general description,
    1) describe the ego driving behavior and the rational behind;
    2) describe the environment (e.g., scene type, time of day, weather condition, road condition)
    that could influence the ego driving behavior;
    3) describe the critical objects (e.g., vehicles, pedestrians, cyclists, traffic lights, traffic signs)
    that could influence the ego driving behavior.

    For the driving difficulity, provide a concise explanation about how difficult it is to drive in the scenario.

    For the noteworthy events, list any events worth noting, with start and end time
    (e.g., ego driving behavior, signs and signals, road user communication, bad behavior and so on).

    Use the exact JSON structure below to present the information.
    If any detail is not visible or cannot be determined, do not guess or assume, simply omit it.
    Required JSON Format:
    [
        {
            "description": "[General description of the video]"
        },
        {
            "driving difficulity explanation": "[A short explanation of driving difficulity]"
        },
        {
            "notice": "[A list of anything notable]"
        }
    ]

    """

    data = {}
    data["system_prompt"] = (
        "You are a professional video annotator tasked with providing a detailed and "
        "structured analysis of the video content."
    )
    data["user_prompt"] = prompt_v1
    data["response"] = annotation
    data["success"] = "True"
    data["prompt_version"] = "v1"
    data["timestamp"] = "N/A"
    return data


def extract_core_info(annotation_file: str, js: dict) -> dict:
    data = {}

    # global description
    data["description"] = js["instances"][2]["attributes"][0]["name"]

    # driving difficulity
    data["driving difficulity explanation"] = js["instances"][4]["attributes"][0]["name"]
    if js["instances"][4]["attributes"][0]["name"] == "":
        print(f"{os.path.basename(annotation_file)} | difficulity info missing")

    # event (note: not every annotation with this info)
    data["notice"] = []
    if len(js["instances"]) > 9:
        idx = 8  # skip other info

        # some labeled with "event_count", some missing
        if js["instances"][idx]["className"] == "event_count":
            idx += 1

        # process each event
        while idx < len(js["instances"]) and js["instances"][idx]["className"] == "start_time":
            tmp = {}
            time_s = js["instances"][idx]["attributes"][0]["name"]
            time_e = js["instances"][idx + 1]["attributes"][0]["name"]
            key = f"Between time {time_s}s and {time_e}s"
            val = js["instances"][idx + 3]["attributes"][0]["name"]

            if time_s == "" or time_e == "" or val == "":
                print(f"{os.path.basename(annotation_file)} | event info missing")
            else:
                tmp[key] = val
                data["notice"].append(tmp)

            idx += 6  # each event spans 6 instances

    return data


def main(args: argparse.Namespace) -> None:
    dir_annotations = args.dir_annotations
    dir_captions = args.dir_captions
    dir_clips = args.dir_clips
    os.makedirs(dir_captions, exist_ok=True)
    annotation_files = glob.glob(os.path.join(dir_annotations, "*.json"))

    for annotation_file in annotation_files:
        # skip annotations without clips
        clip_file = os.path.join(dir_clips, os.path.basename(annotation_file)[:-4] + "camera_front_wide_120fov.mp4")
        if not os.path.exists(clip_file):
            continue

        with open(annotation_file, "r") as f:
            js = json.load(f)

        # skip invalid annotations
        if len(js["instances"]) < 3 or js["instances"][2]["attributes"][0]["name"] == "":
            print(f"{os.path.basename(annotation_file)} | no annotation | size: {os.path.getsize(annotation_file)}")
            continue

        # extract core info from human annotations
        annotation = extract_core_info(annotation_file, js)

        # compose captions for ds-r1 reasoning
        caption = compose_caption(annotation)
        caption_file = os.path.join(dir_captions, os.path.basename(annotation_file))
        with open(caption_file, "w") as f:
            json.dump(caption, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_annotations", type=str, default="/home/xiaodongy/research/data/av_reasoning/annotations")
    parser.add_argument("--dir_captions", type=str, default="/home/xiaodongy/research/data/av_reasoning/captions")
    parser.add_argument("--dir_clips", type=str, default="/home/xiaodongy/research/data/av_reasoning/clips")
    args = parser.parse_args()
    main(args)
