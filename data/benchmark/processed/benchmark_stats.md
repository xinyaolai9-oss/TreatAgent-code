# Benchmark Statistics

## Extraction

| Statistic | Value |
|---|---:|
| raw_rows | 17614 |
| parse_errors | 0 |
| single_disease_single_smiles_rows | 5143 |
| filtered_disease_rows | 1185 |
| kept_rows | 3958 |
| label_counts | {"0": 1760, "1": 2198} |

## Pair-Level Deduplication

| Statistic | Value |
|---|---:|
| input_rows | 3958 |
| valid_canonical_smiles_rows | 3958 |
| invalid_smiles_rows | 0 |
| unique_pair_candidates | 2455 |
| pair_rows | 2160 |
| conflict_pairs_removed | 295 |
| label_counts | {"0": 1054, "1": 1106} |
| unique_drugs | 828 |
| unique_diseases | 686 |

## Splits

### random

| Split | Rows | Labels | Unique Drugs | Unique Diseases |
|---|---:|---|---:|---:|
| train | 1512 | {"1": 774, "0": 738} | 669 | 567 |
| val | 216 | {"0": 105, "1": 111} | 156 | 168 |
| test | 432 | {"1": 221, "0": 211} | 270 | 252 |

Drug overlap: `{"train_val": 101, "train_test": 156, "val_test": 62}`

### drug_disjoint

| Split | Rows | Labels | Unique Drugs | Unique Diseases |
|---|---:|---|---:|---:|
| train | 1512 | {"1": 745, "0": 767} | 362 | 594 |
| val | 216 | {"0": 96, "1": 120} | 193 | 145 |
| test | 432 | {"0": 191, "1": 241} | 273 | 237 |

Drug overlap: `{"train_val": 0, "train_test": 0, "val_test": 0}`


## Temporal Split

| Statistic | Value |
|---|---:|
| pair_rows | 2160 |
| dated_rows | 2160 |
| undated_rows | 0 |
| date_source_counts | {"study_first_submit_date": 1663, "primary_completion_date": 469, "completion_date": 28} |

| Split | Rows | Labels | Date Range |
|---|---:|---|---|
| train | 1512 | {"0": 765, "1": 747} | 1999-11-01 to 2013-04-11 |
| val | 216 | {"1": 114, "0": 102} | 2013-04-16 to 2015-07-16 |
| test | 432 | {"0": 187, "1": 245} | 2015-07-24 to 2026-12-31 |

## Temporal Split: Study First Submit Only

| Statistic | Value |
|---|---:|
| date_policy | study_first_submit |
| pair_rows | 2160 |
| dated_rows | 2160 |
| undated_rows | 0 |
| date_source_counts | {"study_first_submit_date": 2160} |

| Split | Rows | Labels | Date Range |
|---|---:|---|---|
| train | 1512 | {"0": 761, "1": 751} | 1999-11-01 to 2012-09-03 |
| val | 216 | {"0": 98, "1": 118} | 2012-09-04 to 2014-02-27 |
| test | 432 | {"0": 195, "1": 237} | 2014-02-28 to 2019-03-20 |
