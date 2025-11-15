from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from typing import Optional

class StudentModel:
    def __init__(self, model_name: str, model_path: str, device: str = "auto", system_prompt: str = None):
        """
        Initialize local chat model
        
        Args:
            model_path: Path to the model
            device: Device setting ("auto", "cuda", "cpu")
            system_prompt: Optional system message to set model behavior
        """
        self.model_path = model_path
        self.device = device
        self.system_prompt = system_prompt
        
        # Load tokenizer and model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast = False if self.model_name.startswith("yi") else True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True, 
            torch_dtype='auto'
        ).eval()

    def chat(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.95,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Chat function - send a message and get model response
        
        Args:
            message: User input message
            system_prompt: System message for this conversation (optional, uses class system_prompt if not provided)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability
            **kwargs: Additional generation parameters
            
        Returns:
            str: Model response
        """
        # Build messages with optional system message
        messages = []

        if self.model_name.startswith("qwen"):
            return self.model.chat(self.tokenizer, prompt, system=system_prompt, history=None)[0]
        
        # Add system message if provided (either as parameter or class attribute)
        sys_msg = system_prompt if system_prompt is not None else self.system_prompt
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors='pt'
        )
        
        # Move input to the same device as model
        input_ids = input_ids.to(self.model.device)
        
        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )
        
        # Decode output (only the newly generated part)
        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return response
    
    def release(self):
        """
        Release all model resources including GPU memory and model instances.
        This method should be called when the model is no longer needed to free up memory.
        """
        if hasattr(self, 'model'):
            # Delete the model instance and clear GPU memory cache
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Free up GPU memory
        
        if hasattr(self, 'tokenizer'):
            # Delete the tokenizer instance
            del self.tokenizer


# Usage example
if __name__ == "__main__":
    # Initialize model with system message
    model = StudentModel(
        'qwen',
        # '/data1/liqw/checkpoint_out/internlm2_7b/v2-20250627-231247/checkpoint-54-merged'
        '/data1/liqw/checkpoint_out/qwen1_8b/v0-20250627-152837/checkpoint-54-merged'
        # '/data1/liqw/checkpoint_out/yi6b/v0-20250627-153109/checkpoint-54-merged'
    )

    system = "你是一名正在学习汉语知识的学生。"
    query = '''阅读以下材料，选出唯一正确答案并将选项填写到\boxed{}中。

# 材料：

赌：由“赌”构成的词语，《大纲》中只有“赌博”一词，为六级词。  

·熟读下列句子，体会画线词语的意思。  

$       extcircled{1}$ 她的丈夫是个<u>赌棍</u> —一个专靠赌博吃饭的人。  
$       extcircled{2}$ 他们径直朝那幢高大的白色大楼走去，那里尽是渴望赚钱的<u>赌徒</u>。  
$       extcircled{3}$ 妈妈苦口婆心地嘱咐他：“以后可不要再<u>赌</u>了，要知道赌钱不是正经人所为。”  
$       extcircled{4}$ 生活对于雄心勃勃的人是一场精彩绝伦的<u>赌博</u>，需要他用尽全部的智慧、精力和勇气。  


# 问题：

·为下列句子中的“打赌”选择意思最接近的解释。  

$       extcircled{1}$ 我用脑袋<u>打赌</u>，你说谎！
$       extcircled{2}$ 她没有发疯，也不是傻子，她是真诚的。我可以用我的生命<u>打赌</u>。  

A. 为了贏，舍得花大钱
B. 因为怕输，不敢下赌注
C. 与别人对赌谁输谁赢
'''
    
    # Override system message for specific call
    response = model.chat(
        query, 
        temperature=0.8, 
        max_new_tokens=4096,
        system_prompt = system
    )
    print(response)
    
