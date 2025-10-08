import os
import sys
import json

# Add external/KernelBench to path to import modules as a package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_KERNELBENCH = os.path.join(SCRIPT_DIR, "..", "external", "KernelBench")
sys.path.insert(0, EXTERNAL_KERNELBENCH)

from datasets import load_dataset
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template


def generate_json_per_level(output_dir: str = SCRIPT_DIR):
    """
    Load the KernelBench dataset and generate JSON with prompts for all problems,
    saving to three files (one per level) named as kernelbench-cuda-level{level}-all-prompt.json.
    
    Args:
        output_dir: Directory to save the output JSON files
    """
    print("Loading KernelBench dataset from HuggingFace...")
    dataset = load_dataset("ScalingIntelligence/KernelBench")
    
    problems_by_level = {}

    # Process each level
    for level in [1, 2, 3]:
        level_key = f"level_{level}"
        problems_by_level[level] = []

        if level_key not in dataset:
            print(f"Warning: {level_key} not found in dataset, skipping...")
            continue

        curr_level_dataset = dataset[level_key]
        print(f"\nProcessing Level {level}: {len(curr_level_dataset)} problems")

        # Iterate through all problems in this level
        for idx, problem in enumerate(curr_level_dataset):
            problem_id = problem["problem_id"]
            problem_name = problem["name"]
            ref_arch_src = problem["code"]

            # Generate the prompt that would be sent to the LLM
            custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)

            # Create the JSON entry
            problem_entry = {
                "level": level,
                "problem_id": problem_id,
                "backend": "cuda",
                "data_source": "kernelbench",
                "input": custom_cuda_prompt,
                "ref_arch_src": ref_arch_src
            }

            problems_by_level[level].append(problem_entry)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(curr_level_dataset)} problems")

        print(f"Completed Level {level}: {len(curr_level_dataset)} problems processed")

    # Save each level to its own JSON file
    for level in [1, 2, 3]:
        problems = problems_by_level.get(level, [])
        output_file = os.path.join(
            output_dir, f"kernelbench-cuda-level{level}-all-prompt.jsonl"
        )
        print(f"\nSaving {len(problems)} problems to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(problems, f, indent=2)
        print(f"Successfully saved dataset to {output_file}")
        print(f"Level {level} total problems: {len(problems)}")

    return problems_by_level

def add_ref_arch_src_to_orgex_jsonl(level123_dir):
    """
    Loads 'kernelbench-level{level}-orgex-all-all-prompts-only.jsonl' for levels 1, 2, 3 from the given directory,
    adds the 'ref_arch_src' field to each entry by loading the KernelBench dataset,
    and overwrites each file with the updated entries.
    """
    print("\nLoading KernelBench dataset to get ref_arch_src...")
    dataset = load_dataset("ScalingIntelligence/KernelBench")

    # Build a lookup dictionary from the dataset
    # Key: (level, problem_id) -> ref_arch_src
    dataset_lookup = {}
    for level in [1, 2, 3, 4]:
        level_key = f"level_{level}"
        if level_key not in dataset:
            continue
        curr_level_dataset = dataset[level_key]
        for problem in curr_level_dataset:
            problem_id = problem["problem_id"]
            ref_arch_src = problem["code"]
            dataset_lookup[(level, problem_id)] = ref_arch_src

    print(f"Built lookup table with {len(dataset_lookup)} entries")

    for level in [1, 2, 3]:
        orgex_filename = f"kernelbench-level{level}-orgex-all-all-prompts-only.jsonl"
        orgex_path = os.path.join(level123_dir, orgex_filename)
        if not os.path.exists(orgex_path):
            print(f"File not found: {orgex_path} (skipping level {level})")
            continue

        # Load orgex prompts (JSONL format - one JSON object per line)
        orgex_data = []
        with open(orgex_path, "r") as f:
            try:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        orgex_data.append(json.loads(line))
            except Exception as e:
                print(f"Failed to load {orgex_path}: {e}")
                continue

        updated = 0
        not_found = 0
        for entry in orgex_data:
            if "ref_arch_src" not in entry or entry.get("ref_arch_src") is None:
                # Try to find the ref_arch_src from the dataset
                entry_level = entry.get("level", level)  # fallback to current level if not present
                problem_id = entry.get("problem_id")

                if entry_level is not None and problem_id is not None:
                    key = (entry_level, problem_id)
                    if key in dataset_lookup:
                        entry["ref_arch_src"] = dataset_lookup[key]
                        updated += 1
                    else:
                        print(f"Warning: Could not find ref_arch_src for level={entry_level}, problem_id={problem_id}")
                        entry["ref_arch_src"] = None
                        not_found += 1
                else:
                    print(f"Warning: Entry missing level or problem_id: {entry.keys()}")
                    entry["ref_arch_src"] = None
                    not_found += 1

        # Overwrite the file with updated entries (JSONL format - one JSON object per line)
        with open(orgex_path, "w") as f:
            for entry in orgex_data:
                f.write(json.dumps(entry) + "\n")

        print(f"\nAdded 'ref_arch_src' to {updated} entries in {orgex_path}.")
        if not_found > 0:
            print(f"Warning: {not_found} entries could not be matched with dataset")



if __name__ == '__main__':
    generate_json_per_level(SCRIPT_DIR)
    add_ref_arch_src_to_orgex_jsonl(SCRIPT_DIR)