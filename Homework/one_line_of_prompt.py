from transformers import AutoTokenizer, AutoModelForCausalLM
from deep_translator import GoogleTranslator


# 문장/단어 단위를 어떻게 쪼개서 해석할 건지
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 모델 로드: CPU 또는 자동 장치 매핑
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
    )

# 프롬프트 명령
prompt = "Answer as a very kind tour guide."

# 입력 메시지
input_text = "I'm going to travel to Korea, where should I go?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)  # 모델의 장치로 입력 이동

# 텍스트 생성
outputs = model.generate(**inputs, max_length=100)  # 답변 길이는 100으로 제한
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 응답을 한국어로 번역
translated_response = GoogleTranslator(source="en", target="ko").translate(response)

# 출력
print("Prompt:\n", input_text)
print("\nResponse (English):\n", response)
print("\nResponse (Korean):\n", translated_response)  # 번역 결과 출력
