from src.prompts.prompt_formatters import convert_to_training_conversation
from unsloth.trainer import UnslothVisionDataCollator

class CustomVQADataCollator(UnslothVisionDataCollator):
    def __init__(self, *args, template_name="", **kwargs):
        super().__init__(*args, **kwargs)
        self.template_name = template_name

    def __call__(self, examples):
        if self.template_name:
            for ex in examples:
                if "question" in ex and "image" in ex:
                    template = [{"role": "user", "content": [
                        {"type": "image", "image": ex["image"]},
                        {"type": "text", "text": ex["question"]}
                    ]}]
                    ex["messages"] = self.processor.apply_chat_template(
                        template,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

        # Call the original __call__
        return super().__call__(examples)


# class CustomVQADataCollator:
#     def __init__(self, tokenizer, template_name="vivqa.jinja", device="cuda"):
#         self.tokenizer = tokenizer
#         self.template_name = template_name
#         self.device = device

#     def __call__(self, batch):    
#         formatted_texts = []
#         for item in batch:
#             if isinstance(item, dict):
#                 pil_image = item.get("image")
#                 converted_item = convert_to_training_conversation(item, template_name=self.template_name)
#                 print(converted_item)
#                 input_text = self.tokenizer.apply_chat_template(converted_item, add_generation_prompt=True) 
#                 print(input_text)
#                 inputs = self.tokenizer(
#                     pil_image,
#                     input_text,
#                     padding=True,
#                     truncation=True,
#                     add_special_tokens=False,
#                     return_tensors="pt",
#                 ).to(self.device)
#                 formatted_texts.append(inputs)
#             else:
#                 raise ValueError("Each item in the batch must be a dictionary.")    

#         return formatted_texts