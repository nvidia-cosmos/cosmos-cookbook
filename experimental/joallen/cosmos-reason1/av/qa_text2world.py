"""Prompt templates used for the generating question-answer pairs from captions."""

system_template_v0 = """You are an expert in crafting questions to evaluate an text-to-video generation model's \
ability, with a focus on applications in Physical AI such as robots and autonomous vehicles. Your role is to \
generate high-quality, contextually relevant question-answer pairs derived from detailed video captions.
"""


def user_template_fn_v0(caption: str) -> str:
    return f"""I have a text-to-video generation model that takes a text input to produce videos. \
    To rigorously evaluate this model's capability in physical plausibility, I need your help to design binary \
    Visual Question Answering (VQA) pairs. These VQA pairs will be used by evaluators to assess the generated videos.

    Evaluation Objective: The goal is to assess whether the generated videos:
    - Accurately follow the input text instruction.
    - Adhere to physical plausibility and commonsense reasoning.

    Evaluation Method: I will provide you with a video text prompt. Based on the caption, you will propose several \
    binary (Yes/No) VQA questions and the correct answer (Yes/No). Evaluators will only have access to the generated \
    video and the text prompt during evaluation. \
    If all questions are correctly answered, the generated video is considered high-quality and physically plausible.

    IMPORTANT GUIDELINES:
    1. Make questions HIGHLY SPECIFIC with clear visual indicators to check - never ask vague questions
    2. Mention specific objects, colors, positions, or actions that would be easy to verify visually
    3. Avoid subjective judgments like "realistic" or "correct" without specific criteria
    4. For physics questions, focus on observable facts rather than interpretations
    5. Refer to exact details mentioned in the caption whenever possible
    6. Make the questions easy to answer by looking at a single frame or a short sequence

    Suggested Topics for VQA Pairs:

    - Space: Relationship: Generate videos with correct spatial relationships between objects in the scene. \
    Perspective accuracy is important.
    - Space: Interaction: Generate realistic, plausible interactions between objects (including humans and animals).
    - Space: Geometry: Generate videos where object shape, illumination, and occlusions follow the laws of 3D geometry.
    - Time: Actions:
    - - Generate videos that accurately follow specified instructions/actions (movement, direction, etc.).
    - - The motion of subjects should adhere to their physical and kinematic constraints.
    - Time: Order: Generate videos depicting events in the correct chronological sequence \
    (e.g., a robot pushing a box causing the box to slide).
    - Time: Camera: Generate videos that follow specified camera movements, angles, positions, and scene transitions.
    - Physics: Attributes: Generate videos featuring objects that match their specified semantic descriptions, sizes, \
    colors, materials, masses, solidity, etc.
    - Physics: States: Generate videos where object states realistically change according to physical laws (e.g., \
    raw egg cooking into an omelet, ice melting into water, clothes being folded), consistent with the text prompt.
    - Physics: Object Permanence:
    - - Generate videos maintaining consistent object presence without unexplained appearances or disappearances.
    - - Generate realistic breaking or separation of objects following physical laws \
    (e.g., potato cutting, glass breaking).
    - - Generate plausible object collisions and interactions consistent with physical reality (e.g., car collisions, \
    balls bouncing, milk mixing into coffee).

    Task: For each of the nine physical AI topics above, please propose at least two clear, focused binary (Yes/No) \
    VQA questions, including the expected answers based solely on the generated video and provided text prompt.

This is the text prompt:
{caption}
"""


def user_template_fn_v1(caption: str) -> str:  # raw
    return f"""I have a text-to-video generation model that takes a text input to produce videos. \
    To rigorously evaluate this model's capability in physical plausibility, I need your help to design binary \
    Visual Question Answering (VQA) pairs. These VQA pairs will be used by evaluators to assess the generated videos.

    Evaluation Objective: The goal is to assess whether the generated videos:
    - Accurately follow the input text instruction.
    - Adhere to physical plausibility and commonsense reasoning.

    Evaluation Method: I will provide you with a video text prompt. Based on the caption, you will propose several \
    binary (Yes/No) VQA questions and the correct answer (Yes/No). Evaluators will only have access to the generated \
    video and the text prompt during evaluation. \
    If all questions are correctly answered, the generated video is considered high-quality and physically plausible.

    Suggested Topics for VQA Pairs:

    - Space: Relationship: Generate videos with correct spatial relationships between objects in the scene. \
    Perspective accuracy is important.
    - Space: Interaction: Generate realistic, plausible interactions between objects (including humans and animals).
    - Space: Geometry: Generate videos where object shape, illumination, and occlusions follow the laws of 3D geometry.
    - Time: Actions:
    - - Generate videos that accurately follow specified instructions/actions (movement, direction, etc.).
    - - The motion of subjects should adhere to their physical and kinematic constraints.
    - Time: Order: Generate videos depicting events in the correct chronological sequence \
    (e.g., a robot pushing a box causing the box to slide).
    - Time: Camera: Generate videos that follow specified camera movements, angles, positions, and scene transitions.
    - Physics: Attributes: Generate videos featuring objects that match their specified semantic descriptions, sizes, \
    colors, materials, masses, solidity, etc.
    - Physics: States: Generate videos where object states realistically change according to physical laws (e.g., \
    raw egg cooking into an omelet, ice melting into water, clothes being folded), consistent with the text prompt.
    - Physics: Object Permanence:
    - - Generate videos maintaining consistent object presence without unexplained appearances or disappearances.
    - - Generate realistic breaking or separation of objects following physical laws \
    (e.g., potato cutting, glass breaking).
    - - Generate plausible object collisions and interactions consistent with physical reality (e.g., car collisions, \
    balls bouncing, milk mixing into coffee).

    Task: For each of the nine physical AI topics above, please propose at least two clear, focused binary (Yes/No) \
    VQA questions, including the expected answers based solely on the generated video and provided text prompt.

    Try to maintain roughly equal distributions of YES and NO answers.

This is the caption:
{caption}
"""


output_format_v0 = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question 1 (Space: Relationship)": {"type": "string"},
                "answer 1": {"type": "string", "enum": ["YES", "NO"]},
                "question 2 (Space: Relationship)": {"type": "string"},
                "answer 2": {"type": "string", "enum": ["YES", "NO"]},
                "question 3 (Space: Interaction)": {"type": "string"},
                "answer 3": {"type": "string", "enum": ["YES", "NO"]},
                "question 4 (Space: Interaction)": {"type": "string"},
                "answer 4": {"type": "string", "enum": ["YES", "NO"]},
                "question 5 (Space: Geometry)": {"type": "string"},
                "answer 5": {"type": "string", "enum": ["YES", "NO"]},
                "question 6 (Space: Geometry)": {"type": "string"},
                "answer 6": {"type": "string", "enum": ["YES", "NO"]},
                "question 7 (Time: Actions)": {"type": "string"},
                "answer 7": {"type": "string", "enum": ["YES", "NO"]},
                "question 8 (Time: Actions)": {"type": "string"},
                "answer 8": {"type": "string", "enum": ["YES", "NO"]},
                "question 9 (Time: Order)": {"type": "string"},
                "answer 9": {"type": "string", "enum": ["YES", "NO"]},
                "question 10 (Time: Order)": {"type": "string"},
                "answer 10": {"type": "string", "enum": ["YES", "NO"]},
                "question 11 (Time: Camera)": {"type": "string"},
                "answer 11": {"type": "string", "enum": ["YES", "NO"]},
                "question 12 (Time: Camera)": {"type": "string"},
                "answer 12": {"type": "string", "enum": ["YES", "NO"]},
                "question 13 (Physics: Attributes)": {"type": "string"},
                "answer 13": {"type": "string", "enum": ["YES", "NO"]},
                "question 14 (Physics: Attributes)": {"type": "string"},
                "answer 14": {"type": "string", "enum": ["YES", "NO"]},
                "question 15 (Physics: States)": {"type": "string"},
                "answer 15": {"type": "string", "enum": ["YES", "NO"]},
                "question 16 (Physics: States)": {"type": "string"},
                "answer 16": {"type": "string", "enum": ["YES", "NO"]},
                "question 17 (Physics: Object Permanence)": {"type": "string"},
                "answer 17": {"type": "string", "enum": ["YES", "NO"]},
                "question 18 (Physics: Object Permanence)": {"type": "string"},
                "answer 18": {"type": "string", "enum": ["YES", "NO"]},
            },
            "required": [
                "question 1 (Space: Relationship)",
                "answer 1",
                "question 2 (Space: Relationship)",
                "answer 2",
                "question 3 (Space: Interaction)",
                "answer 3",
                "question 4 (Space: Interaction)",
                "answer 4",
                "question 5 (Space: Geometry)",
                "answer 5",
                "question 6 (Space: Geometry)",
                "answer 6",
                "question 7 (Time: Actions)",
                "answer 7",
                "question 8 (Time: Actions)",
                "answer 8",
                "question 9 (Time: Order)",
                "answer 9",
                "question 10 (Time: Order)",
                "answer 10",
                "question 11 (Time: Camera)",
                "answer 11",
                "question 12 (Time: Camera)",
                "answer 12",
                "question 13 (Physics: Attributes)",
                "answer 13",
                "question 14 (Physics: Attributes)",
                "answer 14",
                "question 15 (Physics: States)",
                "answer 15",
                "question 16 (Physics: States)",
                "answer 16",
                "question 17 (Physics: Object Permanence)",
                "answer 17",
                "question 18 (Physics: Object Permanence)",
                "answer 18",
            ],
            "additionalProperties": False,
        },
    },
}

templates = {
    "system_template_v0": system_template_v0,
    "user_template_fn_v0": user_template_fn_v0,
    "output_format_v0": output_format_v0,
}
