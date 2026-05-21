QUESTION_PROMPT = (
    "You are creating visual instruction data for an image model that should handle both "
    "2D seismic reflection images and ordinary non-seismic images. First decide whether "
    "the image appears to be a seismic reflection section. If it is seismic, write one "
    "answerable question about visible seismic patterns such as reflector continuity, "
    "amplitude, geometry, faults, channels, chaos, salt, or transparency. If it is not "
    "seismic, write one answerable question about what the image visibly shows without "
    "forcing a seismic interpretation. "
    "Return only the question."
)

ANSWER_PROMPT_TEMPLATE = (
    "You are a visual interpretation assistant using thinking mode. First decide whether "
    "the image appears to be a 2D seismic reflection section. If it is seismic, answer "
    "using visible seismic evidence such as reflector continuity, amplitude, geometry, "
    "offset, chaos, faults, channels, salt, or transparency. If auxiliary FaultNet "
    "detections are provided, use them as weak evidence, not as guaranteed truth. If no "
    "auxiliary detections are provided, do not assume faults are absent; judge the image "
    "directly. If the image is not seismic, say that it is not a seismic reflection image "
    "and answer plainly using only visible non-seismic content.\n\n"
    "Question: {question}\n"
    "Auxiliary FaultNet detections: {detections}\n\n"
    "Return strict JSON only with keys: reasoning, final_answer. "
    "reasoning must be a list of 2-5 short evidence-grounded steps. "
    "final_answer must be a concise answer to the question. "
    "Do not include unsupported geology, well logs, depth, survey metadata, or exact locations unless visible or supported by detections."
)

K2_ANSWER_PROMPT_TEMPLATE = (
    "The preceding soft visual prefix comes from a Qwen-VL image encoder that processed "
    "an image. First decide whether the image appears to be a 2D seismic reflection "
    "section. If it is seismic, answer the question using visible seismic evidence and "
    "concise geoscience language. If it is not seismic, say that it is not a seismic "
    "reflection image and answer plainly using only visible content. Do not invent labels "
    "or structures.\n\n"
    "Question: {question}\n"
)

MERGE_PROMPT_TEMPLATE = (
    "You are creating the final supervised target for flexible vision-language training. "
    "The model should handle both 2D seismic reflection images and ordinary non-seismic "
    "images. Use the Qwen visual answer as the primary source for what is visible in the "
    "image. Use the K2 geoscience answer only when the image is seismic and only to "
    "improve domain terminology and fluency. FaultNet detections are weak extra evidence, "
    "not guaranteed truth. If FaultNet detections are none, this only means no auxiliary "
    "detector evidence is available; it does not rule out visible faults. If the image is "
    "not seismic, the final answer must say it is not a seismic reflection image and must "
    "describe the visible non-seismic content instead of forcing geoscience language. "
    "Avoid unsupported geology, well logs, depth, survey metadata, exact locations, or "
    "claims not visible in the image.\n\n"
    "Question: {question}\n"
    "Auxiliary FaultNet detections: {detections}\n"
    "Qwen visual reasoning: {qwen_reasoning}\n"
    "Qwen visual answer: {qwen_answer}\n"
    "K2 geoscience reasoning: {k2_reasoning}\n"
    "K2 geoscience answer: {k2_answer}\n\n"
    "Return strict JSON only with keys: reasoning, final_answer. "
    "reasoning must be a list of 2-4 short evidence-grounded steps. "
    "final_answer must be one concise answer using visible evidence. Use seismic domain "
    "language only when the image is actually seismic."
)

IMAGE_PROMPT = (
    "You are encoding an image for a vision-language model. Preserve enough visual "
    "information to decide whether the image is a 2D seismic reflection section or an "
    "ordinary non-seismic image. For seismic images, focus on reflector continuity, "
    "amplitude, geometry, offset, chaos, channels, salt, and transparency. For non-seismic "
    "images, preserve the visible objects, scene, and layout."
)

K2_PROMPT = (
    "The preceding soft visual prefix comes from a Qwen-VL image encoder that processed "
    "an image. First decide whether it appears to be a 2D seismic reflection section. If "
    "it is seismic, classify it using exactly one of: Boring, Bright_Planar, "
    "Bright_Chaotic, Channel, Converging_Amplitudes, Fault, Salt, Transparent_Planar. "
    "Then give one visible seismic reason using reflector continuity, amplitude, "
    "geometry, offset, chaos, or transparency. If it is not seismic, say it is not a "
    "seismic reflection image and describe what it visibly shows. Answer in exactly two "
    "concise sentences. Do not use markdown or bullets."
)

K2_VQA_PROMPT_TEMPLATE = (
    "The preceding soft visual prefix comes from a Qwen-VL image encoder that processed "
    "an image. First decide whether the image appears to be a 2D seismic reflection "
    "section. If it is seismic, answer using visible seismic evidence such as reflector "
    "continuity, amplitude, geometry, offset, chaos, faults, channels, salt, or "
    "transparency. If it is not seismic, say plainly that it is not a seismic reflection "
    "image and answer using the visible non-seismic content. Do not force a seismic "
    "interpretation onto ordinary images.\n\n"
    "Answer as plain text in one concise paragraph. Do not use JSON, arrays, brackets, markdown, "
    "bullets, or separate Reasoning and Final answer sections. Include the visible evidence "
    "inside the same paragraph as the answer.\n\n"
    "Question: {question}\n"
    "Answer:"
)
