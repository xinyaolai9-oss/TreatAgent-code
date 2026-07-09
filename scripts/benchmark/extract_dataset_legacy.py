#!/usr/bin/env python3
"""
Extract JSON data from raw_data.csv
Extracts label, diseases, and smiless fields, filtering for samples with exactly one disease and one SMILES string.
"""

import csv
import json
import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_INPUT_CSV = DATA_DIR / "raw_data.csv"
DEFAULT_OUTPUT_JSON = DATA_DIR / "extracted_single_disease_smiles.json"


def extract_single_disease_single_smiles(input_csv=DEFAULT_INPUT_CSV, output_json=DEFAULT_OUTPUT_JSON):
    """
    Extract data from CSV, keeping only samples with exactly one disease and one SMILES string.
    
    Args:
        input_csv (str): Path to input CSV file
        output_json (str): Path to output JSON file
    """
    extracted_data = []
    total_rows = 0
    filtered_rows = 0
    label_counts = {0: 0, 1: 0}
    
    input_csv = Path(input_csv)
    output_json = Path(output_json)

    with input_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            try:
                # Parse the smiless field from string representation to list
                smiless = ast.literal_eval(row['smiless'])
                diseases = ast.literal_eval(row['diseases'])
                
                # Only keep samples with exactly one disease and one SMILES string
                if len(smiless) == 1 and len(diseases) == 1:
                    # Filter out diseases with punctuation, numbers, etc.
                    disease = diseases[0]
                    disease_lower = disease.lower().strip()
                    if (any(punct in disease for punct in [',', '-', "'", ';', ':', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\']) 
                        or any(char.isdigit() for char in disease) 
                        or disease_lower == 'healthy'):
                        continue
                    
                    # Create extracted data entry
                    entry = {
                        'label': int(row['label']),
                        'disease': disease,      # Take the single disease
                        'smiles': smiless[0]     # Take the single SMILES string
                    }
                    
                    extracted_data.append(entry)
                    filtered_rows += 1
                    label_counts[int(row['label'])] += 1
                    
            except (ValueError, SyntaxError) as e:
                print(f"Skipping row {total_rows} due to parsing error: {e}")
                continue
    
    # Save to JSON file
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {total_rows} total rows")
    print(f"Extracted {filtered_rows} samples with single disease and single SMILES")
    print(f"Label distribution: {label_counts}")
    print(f"Data saved to {output_json}")
    
    return extracted_data

def main():
    input_file = DEFAULT_INPUT_CSV
    output_file = DEFAULT_OUTPUT_JSON
    
    print("Starting extraction of single disease and single SMILES samples...")
    extracted_data = extract_single_disease_single_smiles(input_file, output_file)
    
    # Print summary
    if extracted_data:
        print(f"\n=== Summary ===")
        print(f"Total extracted samples: {len(extracted_data)}")
        
        # Count labels
        label_counts = {}
        for item in extracted_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Label distribution: {label_counts}")
        
        # Sample data
        if extracted_data:
            sample = extracted_data[0]
            print(f"Sample data: label={sample['label']}, disease='{sample['disease']}', smiles='{sample['smiles'][:50]}{'...' if len(sample['smiles']) > 50 else ''}'")

if __name__ == '__main__':
    main()
