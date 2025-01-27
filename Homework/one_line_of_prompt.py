from transformers import AutoTokenizer, AutoModelForCausalLM

# 문장/단어 단위를 어떻게 쪼개서 해석할 건지
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# CPU, GPU, Auto 해보기
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map = "auto",
    offload_folder="offload",
    offload_state_dict=True
    )

# 프롬프트 명령
prompt = "Answer as a very kind tour guide."

# 입력 메시지
input_text = prompt +" I'm going to travel to Korea, where should i go??"
inputs = tokenizer(input_text, return_tensors="pt")

# 텍스트 생성
outputs = model.generate(**inputs, max_length=100)   # 답변 길이는 100으로 제한
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 출력
print("Prompt:\n", input_text)
print("\nResponse:\n", response)