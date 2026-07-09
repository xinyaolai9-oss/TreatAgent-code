# DrugKB Coverage Audit

This audit reports how benchmark drug inputs map to the local DrugCentral-derived DrugKB.

| split | rows | matched | matched rate | unique SMILES | unique matched DrugCentral records | top match methods |
|---|---:|---:|---:|---:|---:|---|
| drug_disjoint_train.json | 1512 | 923 | 0.610 | 362 | 306 | canonical_smiles: 41, drug_name: 747, drug_name_contains: 135, unmatched: 589 |
| drug_disjoint_val.json | 216 | 121 | 0.560 | 193 | 109 | canonical_smiles: 14, drug_name: 85, drug_name_contains: 19, relaxed_smiles: 3, unmatched: 95 |
| drug_disjoint_test.json | 432 | 276 | 0.639 | 273 | 194 | canonical_smiles: 15, drug_name: 192, drug_name_contains: 68, drug_name_fuzzy: 1, unmatched: 156 |
| temporal_submit_train.json | 1512 | 996 | 0.659 | 665 | 444 | canonical_smiles: 54, drug_name: 775, drug_name_contains: 166, relaxed_smiles: 1, unmatched: 516 |
| temporal_submit_val.json | 216 | 112 | 0.518 | 140 | 93 | canonical_smiles: 5, drug_name: 86, drug_name_contains: 21, unmatched: 104 |
| temporal_submit_test.json | 432 | 212 | 0.491 | 242 | 165 | canonical_smiles: 11, drug_name: 163, drug_name_contains: 35, drug_name_fuzzy: 1, relaxed_smiles: 2, unmatched: 220 |
