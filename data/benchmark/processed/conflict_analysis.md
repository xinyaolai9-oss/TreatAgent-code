# Conflict Pair Analysis

## Summary

| Statistic | Value |
|---|---:|
| conflict_pairs | 295 |
| conflict_records | 1365 |
| unique_conflict_drugs | 143 |
| unique_conflict_diseases | 152 |
| case_candidate_count | 150 |

## Recommendation Counts

| Recommendation | Count |
|---|---:|
| inspect_before_reuse | 141 |
| candidate_conflict_case | 76 |
| candidate_temporal_or_status_case | 74 |
| keep_removed | 4 |

## Tag Counts

| Tag | Count |
|---|---:|
| status_driven_conflict | 196 |
| negative_has_stop_reason | 191 |
| same_status_opposite_label | 156 |
| date_overlap | 115 |
| possible_combination_drug_noise | 107 |
| negative_before_positive | 93 |
| positive_before_negative | 87 |
| broad_disease_label | 65 |

## Top Conflict Diseases

| Disease | Count |
|---|---:|
| breast cancer | 17 |
| hiv infection | 10 |
| prostate cancer | 9 |
| multiple myeloma | 9 |
| hypertension | 7 |
| infertility | 6 |
| schizophrenia | 6 |
| rheumatoid arthritis | 6 |
| cancer | 5 |
| lung cancer | 5 |
| asthma | 5 |
| chronic lymphocytic leukemia | 4 |
| cystic fibrosis | 4 |
| major depressive disorder | 4 |
| epilepsy | 3 |

## Candidate Cases

| Disease | Trials | Labels | Tags | Recommendation |
|---|---:|---|---|---|
| schizophrenia | 23 | {"0": 15, "1": 8} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| cystic fibrosis | 14 | {"0": 6, "1": 8} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| rheumatoid arthritis | 13 | {"1": 11, "0": 2} | same_status_opposite_label, date_overlap | candidate_conflict_case |
| schizophrenia | 10 | {"1": 7, "0": 3} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| attention deficit hyperactivity disorder | 9 | {"1": 7, "0": 2} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| major depressive disorder | 9 | {"0": 5, "1": 4} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| systemic lupus erythematosus | 9 | {"1": 4, "0": 5} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| macular degeneration | 8 | {"0": 4, "1": 4} | status_driven_conflict, same_status_opposite_label, date_overlap | candidate_conflict_case |
| osteoarthritis | 8 | {"1": 3, "0": 5} | same_status_opposite_label, date_overlap | candidate_conflict_case |
| amyotrophic lateral sclerosis | 7 | {"1": 1, "0": 6} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, positive_before_negative | candidate_conflict_case |
| psoriasis | 7 | {"1": 5, "0": 2} | same_status_opposite_label, positive_before_negative | candidate_conflict_case |
| anemia | 6 | {"0": 5, "1": 1} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| heart failure | 6 | {"0": 3, "1": 3} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| idiopathic pulmonary fibrosis | 6 | {"0": 5, "1": 1} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| knee osteoarthritis | 6 | {"1": 3, "0": 3} | same_status_opposite_label, date_overlap | candidate_conflict_case |
| leukemia | 6 | {"0": 5, "1": 1} | status_driven_conflict, same_status_opposite_label, negative_has_stop_reason, date_overlap | candidate_conflict_case |
| major depressive disorder | 6 | {"0": 1, "1": 5} | same_status_opposite_label, negative_before_positive | candidate_conflict_case |
| rheumatoid arthritis | 6 | {"1": 5, "0": 1} | same_status_opposite_label, date_overlap | candidate_conflict_case |
| seasonal allergic rhinitis | 6 | {"0": 1, "1": 5} | same_status_opposite_label, negative_before_positive | candidate_conflict_case |
| allergic rhinitis | 5 | {"0": 2, "1": 3} | same_status_opposite_label, negative_before_positive | candidate_conflict_case |
