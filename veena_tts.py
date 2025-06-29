import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "maya-research/veena-tts",
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    llm_int8_enable_fp32_cpu_offload=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts", trust_remote_code=True)
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

def generate_speech(text, speaker="kavya", temperature=0.4, top_p=0.9):
    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN
    ]
    input_ids = torch.tensor([input_tokens], device=model.device)
    max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
        )
    generated_ids = output[0][len(input_tokens):].tolist()
    snac_tokens = [
        token_id for token_id in generated_ids
        if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)
    ]
    if not snac_tokens:
        raise ValueError("No audio tokens generated")
    audio = decode_snac_tokens(snac_tokens)
    return audio

def decode_snac_tokens(snac_tokens):
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        return None
    codes_lvl = [[] for _ in range(3)]
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]
    for i in range(0, len(snac_tokens), 7):
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])
        codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])
    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=snac_model.device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor > 4095)):
            raise ValueError("Invalid SNAC token values")
        hierarchical_codes.append(tensor)
    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)
    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()