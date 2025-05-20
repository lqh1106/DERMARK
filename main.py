import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermarking import generate,detect_robustness
import os
# from wrapper import LMWrapper
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class LMWrapper(torch.nn.Module):
    """A wrapper around the GPT2 model to take ids as input and return logits as output."""

    def __init__(self, path, repetition_penalty=1.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(LMWrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.repetition_penalty = repetition_penalty
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            # self.model = self.model.module

    def forward(self, input_ids):
        # 获取模型的 logits
        outputs = self.model(input_ids)
        logits = outputs.logits  # [batch_size, seq_length, vocab_size]

        if self.repetition_penalty is not None:
            # 获取最后一个时间步的 logits
            logits_last = logits[:, -1, :]  # [batch_size, vocab_size]

            # 构建重复 token 的 mask
            batch_size, seq_length = input_ids.size()
            penalty_mask = torch.zeros_like(logits_last, dtype=torch.bool)  # [batch_size, vocab_size]

            for i in range(batch_size):
                penalty_mask[i, input_ids[i]] = True

            # 应用重复惩罚（张量操作）
            positive_mask = logits_last > 0
            logits_last[penalty_mask & positive_mask] /= self.repetition_penalty
            logits_last[penalty_mask & ~positive_mask] *= self.repetition_penalty

        return logits
    

    
def main():

    model = LMWrapper(model_path)
    model = model.to(model.device)
    
    model.eval()
    vocab_size = model.tokenizer.vocab_size
    
    prompt_set = [model.tokenizer(prompt, return_tensors="pt")]
    batch_size = 1
    batches = [
        [example['input_ids'] for example in prompt_set[i:i + batch_size]]
        for i in range(0, len(prompt_set), batch_size)
    ]

    
    with torch.no_grad():
        

        mark_info = [random.choice([0, 1]) for _ in range(10)] 
        print("mark info",mark_info)
        for batch in batches:
                    
            batch_input = torch.stack(batch, dim=0)[0]
            batch_input = batch_input.to(model.device)
            generated_tokens,N,bit_count,l_t = generate(model,vocab_size, batch_input, mark_info, alpha, max_length ,delta)
            info = detect_robustness(model, batch_input, generated_tokens, vocab_size, alpha, delta,N, bit_count)
            print("\ndetection result:",info)


model_path = '../../Llama-2-7b'
alpha = 0.90            
max_length = 100
delta = 1
prompt = "I am going to tell a story:"

if __name__ == "__main__":

    main()
