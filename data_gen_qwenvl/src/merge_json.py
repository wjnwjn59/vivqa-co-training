import json
import argparse

def merge_json(orig_path, gen_path, paraphr_path, output_path):
    """
    Merges original Q&A, generated questions, and paraphrased questions
    based on image_id into one JSON file with dynamic slot counts.

    Args:
        orig_path (str): Path to the original Q&A JSON file
        gen_path (str): Path to the generated questions JSON file
        paraphr_path (str): Path to the paraphrased questions JSON file
        output_path (str): Path where to save the merged output JSON
    """

    # Load source files
    with open(orig_path, 'r', encoding='utf-8') as f:
        orig_data = json.load(f)
    with open(gen_path, 'r', encoding='utf-8') as f:
        gen_data = json.load(f)
    with open(paraphr_path, 'r', encoding='utf-8') as f:
        paraphr_data = json.load(f)

    # Group original questions & answers by image_id
    qa_map = {}
    for ann in orig_data.get("annotations", {}).values():
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        qa_map.setdefault(img_id, {"questions": [], "answers": []})
        qa_map[img_id]["questions"].append(ann.get("question", ""))
        qa_map[img_id]["answers"].append(ann.get("answer", ""))

    # Group generated (alternate) questions by image_id
    gen_map = {}
    for item in gen_data.values():
        img_id = item.get("image_id")
        if img_id is None:
            continue
        alt = item.get("question_generated", {})
        ordered = [
            alt[k] for k in sorted(
                alt.keys(), key=lambda x: int(x.rsplit("_", 1)[-1])
            )
        ]
        gen_map[img_id] = ordered

    # Group paraphrased questions by image_id
    paraphr_map = {}
    for item in paraphr_data.values():
        img_id = item.get("image_id")
        if img_id is None:
            continue
        text = item.get("question_generated", {})\
                   .get("question_paraphrased", "")
        paraphr_map.setdefault(img_id, []).append(text)

    # Merge into final structure with dynamic slots
    merged = {}
    all_ids = set(qa_map) | set(gen_map) | set(paraphr_map)
    for img_id in sorted(all_ids):
        key = str(img_id)
        merged[key] = {"image_id": img_id}

        # Original questions/answers
        qs = qa_map.get(img_id, {}).get("questions", [])
        ans = qa_map.get(img_id, {}).get("answers", [])
        merged[key]["original_question"] = {
            f"question_{i+1}": q for i, q in enumerate(qs)
        }
        merged[key]["original_answer"] = {
            f"answer_{i+1}": a for i, a in enumerate(ans)
        }

        # Generated questions
        gen_list = gen_map.get(img_id, [])
        gen_q = {f"generated_question_{i+1}": q for i, q in enumerate(gen_list)}

        # Paraphrased questions
        parap_list = paraphr_map.get(img_id, [])
        parap_q = {f"paraphrased_question_{i+1}": p for i, p in enumerate(parap_list)}

        merged[key]["qwenvl_generated"] = {
            "question_paraphrased": parap_q,
            "question_generated": gen_q
        }

    # Write merged JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f"Saved merged data to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge original Q&A, generated, and paraphrased questions by image_id"
    )
    parser.add_argument(
        "--orig", required=True,
        help="Path to original Q&A JSON file"
    )
    parser.add_argument(
        "--gen", required=True,
        help="Path to generated questions JSON file"
    )
    parser.add_argument(
        "--paraphr", required=True,
        help="Path to paraphrased questions JSON file"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output merged JSON file"
    )
    args = parser.parse_args()
    merge_json(
        orig_path=args.orig,
        gen_path=args.gen,
        paraphr_path=args.paraphr,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
