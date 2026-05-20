QUESTION_PROMPT = (
    "You are creating visual instruction data for 2D seismic interpretation. "
    "Look at this seismic image and write one answerable question about visible seismic "
    "patterns. The question must be specific to the image and should ask about reflector "
    "continuity, amplitude, geometry, faults, channels, chaos, or transparency. "
    "Return only the question."
)

ANSWER_PROMPT_TEMPLATE = (
    "You are a seismic interpretation assistant using thinking mode. "
    "Answer the question using only visible evidence from the seismic image. "
    "If auxiliary FaultNet detections are provided, use them as weak evidence, not as guaranteed truth. "
    "If no auxiliary detections are provided, do not assume faults are absent; judge the image directly.\n\n"
    "Question: {question}\n"
    "Auxiliary FaultNet detections: {detections}\n\n"
    "Return strict JSON only with keys: reasoning, final_answer. "
    "reasoning must be a list of 2-5 short evidence-grounded steps. "
    "final_answer must be a concise answer to the question. "
    "Do not include unsupported geology, well logs, depth, survey metadata, or exact locations unless visible or supported by detections."
)

K2_ANSWER_PROMPT_TEMPLATE = (
    "The preceding soft visual prefix comes from a Qwen-VL image encoder that processed "
    "a 2D seismic image. Answer this seismic interpretation question using visible image "
    "evidence only. Use concise geoscience language. Do not invent labels or structures.\n\n"
    "Question: {question}\n"
)

MERGE_PROMPT_TEMPLATE = (
    "You are creating the final supervised target for seismic vision-language training. "
    "Use the Qwen visual answer as the primary source for what is visible in the image. "
    "Use the K2 geoscience answer only to improve domain terminology and fluency when it is provided. "
    "FaultNet detections are weak extra evidence, not guaranteed truth. "
    "If FaultNet detections are none, this only means no auxiliary detector evidence is available; it does not rule out visible faults. "
    "Write an answer that is grounded in the image and avoids unsupported geology, well logs, survey metadata, "
    "exact locations, or claims not visible in the seismic section.\n\n"
    "Question: {question}\n"
    "Auxiliary FaultNet detections: {detections}\n"
    "Qwen visual reasoning: {qwen_reasoning}\n"
    "Qwen visual answer: {qwen_answer}\n"
    "K2 geoscience reasoning: {k2_reasoning}\n"
    "K2 geoscience answer: {k2_answer}\n\n"
    "Return strict JSON only with keys: reasoning, final_answer. "
    "reasoning must be a list of 2-4 short evidence-grounded steps. "
    "final_answer must be one concise seismic interpretation answer using visible evidence and domain language."
)

IMAGE_PROMPT = (
    "You are encoding a seismic reflection image for a geoscience language model. "
    "Focus on reflector continuity, amplitude, geometry, offset, chaos, channels, salt, "
    "and transparency."
)

K2_PROMPT = (
    "The preceding soft visual prefix comes from a Qwen-VL image encoder that processed "
    "a seismic reflection image. Interpret the image. Classify it using exactly one of: "
    "Boring, Bright_Planar, Bright_Chaotic, Channel, Converging_Amplitudes, Fault, Salt, "
    "Transparent_Planar. Answer in exactly two concise sentences. Sentence 1: <Class>. "
    "Sentence 2: one visible seismic reason using reflector continuity, amplitude, "
    "geometry, offset, chaos, or transparency. Do not use markdown or bullets."
)

K2_VQA_PROMPT_TEMPLATE = (
    "The preceding soft visual prefix comes from a Qwen-VL image encoder that processed "
    "a 2D seismic reflection image. Answer the user's seismic interpretation question using "
    "only visible image evidence. Stay in seismic image interpretation; do not discuss "
    "seismographs, earthquakes, stations, papers, conferences, or soil dynamics unless the "
    "image visibly contains that information.\n\n"
    "Use exactly this format:\n"
    "Reasoning:\n"
    "- one short visible seismic clue\n"
    "- one short visible seismic clue\n"
    "Final answer: one concise answer grounded in reflector continuity, amplitude, geometry, "
    "offset, chaos, or transparency.\n\n"
    "Question: {question}\n"
    "Answer:"
)

