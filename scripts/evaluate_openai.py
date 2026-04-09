import os
import random
import csv
import re
from langchain_openai import ChatOpenAI
import hydra
from omegaconf import DictConfig
from pathlib import Path
from io import StringIO

"""
Historical Knowledge Evaluation - OpenAI Models Only
Evaluates different OpenAI models on historical reasoning tasks
Only requires OPENAI_API_KEY
Uses Hydra for configuration and result organization
Supports multiple dataset formats
"""

def load_dataset_format1(file_path):
    """Load original format with placeholders"""
    qa_dataset = []
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    file_content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                file_content = file.read()
            break
        except UnicodeDecodeError:
            continue
    
    if file_content is None:
        raise Exception("Could not decode file")
    
    csv_file = StringIO(file_content)
    reader = csv.reader(csv_file)
    next(reader)  # Skip header
    
    for row in reader:
        if len(row) < 8:
            continue
        
        try:
            template_id, qa_template, qa_id = row[0], row[1], row[2]
            num_placeholders, num_choices = int(row[3]), int(row[4])
            ground_truth = row[5]
            all_choices = [c.strip() for c in row[6].strip().split(";")]
            
            # Randomize MCQ choices
            if num_choices == 4:
                dict_letter_index = {"A": 0, "B": 1, "C": 2, "D": 3}
                ground_truth_index = dict_letter_index[ground_truth]
                indices = [0, 1, 2, 3]
                random.shuffle(indices)
                shuffled_choices = [all_choices[i] for i in indices]
                ground_truth_index_new = indices.index(ground_truth_index)
                dict_index_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
                ground_truth = dict_index_letter[ground_truth_index_new]
                all_choices = shuffled_choices
            
            # Insert placeholders
            for i in range(num_placeholders):
                if 7 + i < len(row):
                    qa_template = qa_template.replace(f"[p{i+1}]", row[7 + i])
            
            # Construct question
            if num_choices == 4:
                prefix = "You are an expert on historical events. Please select the best choice for the following" \
                "multiple choice question. Don't show any reasoning process. Just give me the letter of your best choice.\n"
                suffix = "\nA. " + all_choices[0] + "\nB. " + all_choices[1] + "\nC. " + all_choices[2] + "\nD. " + all_choices[3]
                llm_qa_question = prefix + qa_template + suffix
            elif num_choices == 2 and ground_truth in ["TRUE", "FALSE"]:
                prefix = "You are an expert on historical events. Please answer the following true or false question. \
                Don't show any reasoning process. Just give me TRUE or FALSE as your answer. Here is the question: "
                llm_qa_question = prefix + qa_template
            else:
                continue
            
            qa_dataset.append({
                "template_id": template_id,
                "question_id": qa_id,
                "question_text": llm_qa_question,
                "ground_truth": ground_truth
            })
        except:
            continue
    
    return qa_dataset


def load_dataset_format2(file_path):
    """Load new format with pre-formed questions"""
    qa_dataset = []
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-8-sig']
    file_content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                file_content = file.read()
            break
        except UnicodeDecodeError:
            continue
    
    if file_content is None:
        raise Exception("Could not decode file")
    
    lines = file_content.strip().split('\n')
    i, question_id = 1, 0  # Skip header
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        try:
            row = list(csv.reader([line]))[0]
            if len(row) < 4:
                i += 1
                continue
            
            sn, template, question_text, answer = row[0], row[1], row[2], row[3]
            if not question_text or not answer:
                i += 1
                continue
            
            question_id += 1
            
            # TRUE/FALSE question
            if answer.upper() in ["TRUE", "FALSE"]:
                prefix = "You are an expert on historical events. Please answer the following true or false question. Don't show any reasoning process. Just give me TRUE or FALSE as your answer. Here is the question: "
                llm_qa_question = prefix + question_text
                ground_truth = answer.upper()
            
            # Multiple choice question
            elif answer.upper() in ["A", "B", "C", "D"]:
                choices = []
                i += 1
                while i < len(lines) and len(choices) < 4:
                    next_line = lines[i].strip()
                    if next_line and next_line[0] in ['A', 'B', 'C', 'D'] and next_line[1] == '.':
                        choices.append(next_line[3:].strip())
                        i += 1
                    else:
                        break
                
                if len(choices) != 4:
                    continue
                
                prefix = "You are an expert on historical events. Please select the best choice for the following multiple choice question. Don't show any reasoning process. Just give me the letter of your best choice.\n"
                suffix = "\nA. " + choices[0] + "\nB. " + choices[1] + "\nC. " + choices[2] + "\nD. " + choices[3]
                llm_qa_question = prefix + question_text + suffix
                ground_truth = answer.upper()
            else:
                i += 1
                continue
            
            qa_dataset.append({
                "template_id": sn,
                "question_id": str(question_id),
                "question_text": llm_qa_question,
                "ground_truth": ground_truth
            })
        except:
            i += 1
            continue
        
        i += 1
    
    return qa_dataset


def load_dataset_auto(file_path):
    """Auto-detect format and load"""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
    except:
        with open(file_path, 'r', encoding='latin-1') as f:
            first_line = f.readline()
    
    if 'Template_ID' in first_line or 'Question_ID' in first_line:
        print("📋 Format: Original (with placeholders)")
        return load_dataset_format1(file_path)
    elif 'S.N' in first_line:
        print("📋 Format: New (pre-formed questions)")
        return load_dataset_format2(file_path)
    else:
        raise Exception("Unknown dataset format")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Setup API key
    os.environ.update({"OPENAI_API_KEY": cfg.api_key})
    
    # Get output directory from Hydra
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    print("\n" + "=" * 80)
    print("HISTORICAL KNOWLEDGE EVALUATION - OPENAI MODELS")
    print("=" * 80)
    print(f"📁 Results will be saved to: {output_dir}")
    print("=" * 80)
    
    # Configure models from config file
    models = {}
    print("\n" + "=" * 80)
    print("LOADING OPENAI MODELS")
    print("=" * 80)
    
    for model_name in cfg.models:
        try:
            models[model_name] = {
                "llm": ChatOpenAI(model_name=model_name, temperature=cfg.temperature),
                "name": model_name
            }
            print(f"Loaded {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    if not models:
        print("\n⚠ ERROR: No models loaded!")
        print("Please check your API key and model configuration.")
        exit(1)
    
    print(f"\n✓ Successfully loaded {len(models)} OpenAI model(s)")
    
    def get_model_response(llm, text):
        """Get response from LLM model"""
        try:
            return llm.invoke(input=text).content
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def extract_answer(llm_output, ground_truth):
        """Extract just the letter (A/B/C/D or TRUE/FALSE) from LLM output"""
        llm_output = llm_output.strip()
        
        # For TRUE/FALSE questions
        if ground_truth in ["TRUE", "FALSE"]:
            if "TRUE" in llm_output.upper():
                return "TRUE"
            elif "FALSE" in llm_output.upper():
                return "FALSE"
            else:
                return llm_output  # Return as-is if unclear
        
        # For multiple choice (A/B/C/D)
        # Try to extract just the letter from responses like "A. description" or "The answer is A"
        import re
        
        # Pattern 1: Letter at the start (most common)
        match = re.match(r'^([A-D])[.\s)]', llm_output)
        if match:
            return match.group(1)
        
        # Pattern 2: "The answer is X" or similar
        match = re.search(r'\b([A-D])\b', llm_output)
        if match:
            return match.group(1)
        
        # If no clear pattern, return first character if it's a valid choice
        if len(llm_output) > 0 and llm_output[0] in ['A', 'B', 'C', 'D']:
            return llm_output[0]
        
        # Return as-is if can't extract
        return llm_output
    
    # Load QA dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    qa_dataset = load_dataset_auto(cfg.dataset_path)
    print(f"✓ Loaded {len(qa_dataset)} questions")
    
    # Run evaluation
    accuracy_summary = []
    
    for idx, (model_key, model_info) in enumerate(models.items(), 1):
        print("\n" + "=" * 80)
        print(f"EVALUATING MODEL {idx}/{len(models)}: {model_info['name']}")
        print("=" * 80)
        
        model_results = []
        
        for q_idx, qa_item in enumerate(qa_dataset):
            print(f"Question {q_idx+1}/{len(qa_dataset)}...", end='\r')
            
            resp = get_model_response(model_info["llm"], qa_item["question_text"])
            
            # Extract just the letter/answer
            cleaned_answer = extract_answer(resp, qa_item["ground_truth"])
            
            # Check correctness (now comparing clean answers)
            is_correct = qa_item["ground_truth"].strip().upper() == cleaned_answer.strip().upper()
            
            model_results.append({
                "template_id": qa_item["template_id"],
                "question_id": qa_item["question_id"],
                "llm_output": cleaned_answer,  # Save only the letter
                "ground_truth": qa_item["ground_truth"],
                "correct": "Correct" if is_correct else "Wrong",
                "question_text": qa_item["question_text"]
            })
        
        # Calculate accuracy for this model
        correct = sum(1 for r in model_results if r["correct"] == "Correct")
        total = len(model_results)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n✓ {model_info['name']}: {accuracy:.2%} ({correct}/{total})")
        
        # Save results for this model immediately
        model_file = output_dir / f"results_{model_info['name'].replace('-', '_').replace('.', '_')}.csv"
        with open(model_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['template_id', 'question_id', 
                                                    'llm_output', 'ground_truth', 'correct', 'question_text'])
            writer.writeheader()
            writer.writerows(model_results)
        print(f"💾 Saved: {model_file}")
        
        # Add to accuracy summary
        accuracy_summary.append({
            'rank': idx,  # Will be re-ranked later
            'model_name': model_info['name'],
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        })
        
        # Clear model results from memory
        model_results = None
    
    # Save accuracy summary (re-rank by accuracy)
    print("\n" + "=" * 80)
    print("SAVING SUMMARY")
    print("=" * 80)
    
    accuracy_summary = sorted(accuracy_summary, key=lambda x: x['accuracy'], reverse=True)
    
    summary_file = output_dir / "accuracy_summary.csv"
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['rank', 'model_name', 'correct', 'total', 'accuracy_percent'])
        writer.writeheader()
        for rank, stats in enumerate(accuracy_summary, 1):
            writer.writerow({
                'rank': rank,
                'model_name': stats['model_name'],
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy_percent': f"{stats['accuracy']*100:.2f}%"
            })
    
    print(f"✓ Saved: {summary_file}")
    
    # Display final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS - OPENAI MODELS RANKED")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Model':<25} {'Accuracy':<12} {'Score'}")
    print("-" * 60)
    
    for rank, stats in enumerate(accuracy_summary, 1):
        print(f"{rank:<6} {stats['model_name']:<25} {stats['accuracy']:>10.2%} {stats['correct']}/{stats['total']}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  1. accuracy_summary.csv - Performance summary")
    print(f"  2. results_*.csv - Individual model results ({len(models)} files)")
    print("=" * 80)

if __name__ == "__main__":
    main()