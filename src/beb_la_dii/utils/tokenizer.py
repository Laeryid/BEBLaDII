from transformers import AutoTokenizer
import torch

def get_tokenizer(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Возвращает токенизатор Qwen2.5, настроенный для использования в дистилляции.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Убеждаемся, что специальные токены для рассуждения присутствуют
    special_tokens = ["<|thought|>", "<|im_start|>", "<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Настройка padding токена
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

if __name__ == "__main__":
    # Тест загрузки
    tok = get_tokenizer()
    print(f"Vocab size: {len(tok)}")
    test_str = "<|im_start|>system\nВы - ИИ-помощник.<|im_end|>\n<|im_start|>user\nПривет!<|im_end|>\n<|im_start|>assistant\n<|thought|>\nПривет!"
    tokens = tok.encode(test_str)
    print(f"Encoded length: {len(tokens)}")
    print(f"Decoded: {tok.decode(tokens)}")
