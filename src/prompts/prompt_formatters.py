from jinja2 import Environment, FileSystemLoader
from pathlib import Path

template_dir = Path(__file__).parent / "templates"
env = Environment(loader=FileSystemLoader(template_dir))

def convert_to_training_conversation(sample, template_name="vivqa.jinja"):
    question = sample["question"]
    answers = sample["answers"]
    pil_image = sample["image"]

    if isinstance(answers, list):
        answer = answers[0]
    else:
        answer = answers

    template = env.get_template(template_name)
    instruction = template.render(question=question.strip())

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": pil_image}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer}
            ]
        },
    ]

    return {"messages": conversation}


def convert_to_inference_conversation(image, question, template_name="vivqa.jinja"):
    template = env.get_template(template_name)
    instruction = template.render(question=question.strip())

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
    }
