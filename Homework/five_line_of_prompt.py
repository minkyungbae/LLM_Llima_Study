from transformers import AutoTokenizer, AutoModelForCausalLM
from googletrans import Translator

# 문장/단어 단위를 어떻게 쪼개서 해석할 건지
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# CPU, GPU, Auto 해보기
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
)

# 번역기 초기화
translator = Translator()

# 프롬프트 명령
prompt = "Answer like a tour-oriented tour guide. also you are been working for more than five years. and your personality is kind, organized, and friendly."

# 입력 메시지
input_text = prompt + " I'm going to travel to Korea, where should I go?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)  # 모델의 장치로 입력 이동

# 텍스트 생성
outputs = model.generate(**inputs, max_length=100)  # 답변 길이는 100으로 제한
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 응답을 한국어로 번역
translated_response = translator.translate(response, src="en", dest="ko").text
# 출력
print("Prompt:\n", input_text)
print("\nResponse (English):\n", response)
print("\nResponse (Korean):\n", translated_response)  # 한국어로 번역된 텍스트 출력
