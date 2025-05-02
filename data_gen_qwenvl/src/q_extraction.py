import argparse
import json

def extract_questions_by_image_id(data):
    """
    Extract and organize questions from a dataset by their image_id.
    
    This function takes a JSON structure containing annotations with image_id and question fields,
    groups all questions that belong to the same image, and restructures them into a new format
    with up to three questions per image (filling empty slots with empty strings).
    
    Args:
        data (dict): The input data dictionary containing an 'annotations' key with question data
        
    Returns:
        dict: A dictionary where each key is an image_id (as string) with structured question data
              Format: {
                  "image_id": The image ID (integer),
                  "original_question": {
                      "question_1": "First question for this image",
                      "question_2": "Second question for this image (if exists)",
                      "question_3": "Third question for this image (if exists)"
                  }
              }
    """
    result = {}
    # Temporary dictionary to hold questions grouped by image_id
    temp = {}

    # Step 1: Group all questions by their respective image_id
    for key, value in data["annotations"].items():
        image_id = value["image_id"]
        question = value["question"]

        # Create a new list for this image_id if we haven't seen it before
        if image_id not in temp:
            temp[image_id] = []
        # Add the question to the list for this image_id
        temp[image_id].append(question)

    # Step 2: Format the grouped questions into the required output structure
    for image_id, questions in temp.items():
        question_dict = {}
        # Assign each question to a numbered key (question_1, question_2, etc.)
        for i, q in enumerate(questions, 1):
            question_dict[f"question_{i}"] = q
        
        # Add the formatted entry to the result dictionary using image_id as key
        result[str(image_id)] = {
            "image_id": image_id,
            "original_question": question_dict
        }

    return result

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process input and output JSON files for question extraction')
    parser.add_argument('--input_json', type=str, required=True, 
                        help='Path to the input JSON file with annotations')
    parser.add_argument('--output_json', type=str, required=True, 
                        help='Path to the output JSON file for processed questions')

    # Parse arguments
    args = parser.parse_args()

    # Step 1: Load the input JSON file containing question annotations
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {args.input_json} not found. Please check the file path.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in the input file {args.input_json}.")
        exit(1)

    # Step 2: Process the data to extract and structure questions by image_id
    output_json = extract_questions_by_image_id(input_data)

    # Step 3: Save the processed data to a new JSON file
    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=4)
        print(f"Successfully processed and saved data for {len(output_json)} images.")
    except IOError:
        print(f"Error: Unable to write to output file {args.output_json}. Please check permissions and path.")

if __name__ == "__main__":
    main()
