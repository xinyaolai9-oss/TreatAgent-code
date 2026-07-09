import argparse
import json
import os
import re
import traceback
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from treatagent.config.baselines import get_response
from treatagent.orchestration.argument_graph_scorer import argument_factors_from_result
from treatagent.orchestration.orchestrator import TreatAgentOrchestrator
from treatagent.utils import build_sample_id


def load_json_data(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"Error loading JSON file: {exc}")
        return []


def predict_multiagent(
    formula,
    disease,
    model,
    agent_version="eg",
    label=None,
    sample_id=None,
    threshold=0.5,
    generate_report=False,
    use_memory=False,
    knowledge_cutoff_date=None,
    drug_names=None,
    drug_identifiers=None,
    drug_inchikey=None,
):
    system = TreatAgentOrchestrator(
        model=model,
        agent_version=agent_version,
        generate_report=generate_report,
        use_memory=use_memory,
        knowledge_cutoff_date=knowledge_cutoff_date,
    )
    result = system.analyze(
        formula,
        disease,
        label=label,
        sample_id=sample_id,
        drug_names=drug_names,
        drug_identifiers=drug_identifiers,
        drug_inchikey=drug_inchikey,
    )

    normalized_version = str(agent_version).lower()
    judge = result.get("llm_judge") or {}
    use_llm_judge_score = (
        normalized_version == "full"
        and isinstance(judge, dict)
        and judge.get("treatment_score") is not None
    )
    use_arg_final_score = normalized_version in {"eg", "full"} and not use_llm_judge_score
    if use_llm_judge_score:
        argument_row = argument_factors_from_result(result)
        argument_factors = argument_row.get("factors") or {}
        final_probability = float(judge.get("treatment_score"))
        final_threshold = 0.5 if threshold is None else float(threshold)
        prediction_binary = 1.0 if final_probability >= final_threshold else 0.0
        result["argument_probability"] = round(float(argument_factors.get("raw_argument_score", 0.0)), 6)
        result["argument_prediction"] = 1 if result["argument_probability"] >= _resolve_arg_threshold(None) else 0
        result["argument_factors"] = argument_factors
        result["llm_judge_probability"] = round(final_probability, 6)
        result["llm_judge_prediction"] = int(prediction_binary)
        result["final_score_source"] = "llm_evidence_judge"
        result["final_threshold"] = final_threshold
    elif use_arg_final_score:
        argument_row = argument_factors_from_result(result)
        argument_factors = argument_row.get("factors") or {}
        final_probability = float(argument_factors.get("raw_argument_score", 0.0))
        final_threshold = _resolve_arg_threshold(threshold)
        prediction_binary = 1.0 if final_probability >= final_threshold else 0.0
        result["argument_probability"] = round(final_probability, 6)
        result["argument_prediction"] = int(prediction_binary)
        result["argument_factors"] = argument_factors
        result["final_score_source"] = "argument_graph"
        result["final_threshold"] = final_threshold
    else:
        calibrated_probability = result["calibration"]["calibrated_probability"]
        final_probability = calibrated_probability
        final_threshold = 0.5 if threshold is None else threshold
        prediction_binary = 1.0 if final_probability >= final_threshold else 0.0
        result["final_score_source"] = "llm_synthesis_calibration"
        result["final_threshold"] = final_threshold

    print(f"Raw score: {result['synthesis']['raw_score']:.4f}/10")
    print(f"Calibrated probability: {result['calibration']['calibrated_probability']:.4f}")
    if use_llm_judge_score:
        print(f"LLM judge probability: {final_probability:.4f}")
    if use_arg_final_score:
        print(f"ARG probability: {final_probability:.4f}")
    print(f"Final score source: {result['final_score_source']}")
    print(f"Decision threshold: {final_threshold:.4f}")
    return {
        "prediction_binary": prediction_binary,
        "analysis": result,
    }


def _resolve_arg_threshold(threshold):
    if threshold is not None:
        return float(threshold)
    raw = os.getenv("TREATAGENT_ARG_THRESHOLD", "0.36")
    try:
        return float(raw)
    except ValueError:
        return 0.36


def _extract_binary_answer(response_text):
    match = re.search(r"ANSWER:\s*([01])", response_text.upper())
    if match:
        return int(match.group(1))

    fallback_match = re.search(r"\b([01])\b", response_text)
    if fallback_match:
        return int(fallback_match.group(1))

    print("Warning: Could not extract valid score from response, defaulting to 0")
    return 0


def predict_direct(formula, disease, model):
    prompt = f"""
    Task: Return ONLY a single digit: 0 (impossible) or 1 (possible).
    Format your response as: ANSWER: 0 or ANSWER: 1

    Examples:
    Can '[H][C@]12C[C@@H](O)C=C[C@]11CCN(C)CC3=C1C(O2)=C(OC)C=C3' cure 'alzheimer disease'?
    ANSWER: 1

    Can {formula} cure {disease}?
    ANSWER:
    """
    response = get_response(prompt, model)
    print("response: ", response)
    score = _extract_binary_answer(response.strip())
    print("direct score:", score)
    return float(score)


def predict_cot(formula, disease, model):
    prompt = f"""
    You are a pharmaceutical expert. Please analyze whether the smiles {formula} can cure {disease} through systematic reasoning.
    Your response MUST follow this exact format:

    ANALYSIS:
    [Let's think step by step.]

    CONCLUSION:
    Based on the above analysis, can {formula} cure {disease}?
    ANSWER: [0 or 1]
    Where 0 = cannot cure, 1 = can potentially cure

    Can {formula} cure {disease}?
    """
    response = get_response(prompt, model)
    print("response: ", response)
    score = _extract_binary_answer(response.strip())
    print("direct score:", score)
    return float(score)


def _build_rag_context(analysis):
    evidence_graph = analysis.get("evidence_graph") or {}
    expert_outputs = analysis.get("expert_outputs") or {}
    lines = [
        f"Drug SMILES: {analysis.get('drug') or evidence_graph.get('drug')}",
        f"Disease: {analysis.get('disease') or evidence_graph.get('disease')}",
        "",
        "Retrieved evidence:",
    ]

    for expert_name in ["DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"]:
        output = expert_outputs.get(expert_name) or {}
        status = output.get("status", "missing")
        lines.append(f"\n[{expert_name}] status={status}")
        evidence_items = output.get("evidence") or []
        if not evidence_items:
            lines.append("- No retrieved evidence.")
            continue
        for item in evidence_items[:8]:
            claim = str(item.get("claim", "")).strip()
            value = item.get("value")
            impact = item.get("impact")
            confidence = item.get("confidence")
            source = item.get("source")
            lines.append(
                f"- claim={claim}; value={value}; impact={impact}; confidence={confidence}; source={source}"
            )

    return "\n".join(lines)


def predict_rag(
    formula,
    disease,
    model,
    label=None,
    sample_id=None,
    knowledge_cutoff_date=None,
    drug_names=None,
    drug_identifiers=None,
    drug_inchikey=None,
):
    retriever = TreatAgentOrchestrator(
        model="local",
        agent_version="eg",
        generate_report=False,
        use_memory=False,
        knowledge_cutoff_date=knowledge_cutoff_date,
    )
    analysis = retriever.analyze(
        formula,
        disease,
        label=label,
        sample_id=sample_id,
        drug_names=drug_names,
        drug_identifiers=drug_identifiers,
        drug_inchikey=drug_inchikey,
    )
    context = _build_rag_context(analysis)
    prompt = f"""
You are a conservative pharmaceutical evidence reviewer.

Task:
Use ONLY the retrieved evidence below to decide whether the drug could treat the disease.
Do not use hidden prior knowledge beyond the provided evidence.
Do not use EvidenceGraph scores, reliability weights, calibration, or TreatAgent's final score.

Return this exact format:
ANALYSIS:
[brief evidence-grounded reasoning]

ANSWER: 0 or 1

Where:
0 = evidence does not support therapeutic potential
1 = evidence supports therapeutic potential

Retrieved evidence:
{context}
"""
    response = get_response(prompt, model)
    print("response: ", response)
    score = _extract_binary_answer(response.strip())
    print("rag score:", score)
    return {
        "prediction_binary": float(score),
        "analysis": analysis,
        "rag_context": context,
        "rag_response": response,
    }


def predict_therapeutic_potential(
    formula,
    disease,
    method="multiagent",
    model="gpt-4o",
    agent_version="eg",
    label=None,
    sample_id=None,
    threshold=0.5,
    generate_report=False,
    use_memory=False,
    knowledge_cutoff_date=None,
    drug_names=None,
    drug_identifiers=None,
    drug_inchikey=None,
):
    if method == "multiagent":
        return predict_multiagent(
            formula,
            disease,
            model,
            agent_version=agent_version,
            label=label,
            sample_id=sample_id,
            threshold=threshold,
            generate_report=generate_report,
            use_memory=use_memory,
            knowledge_cutoff_date=knowledge_cutoff_date,
            drug_names=drug_names,
            drug_identifiers=drug_identifiers,
            drug_inchikey=drug_inchikey,
        )
    if method == "direct":
        return predict_direct(formula, disease, model)
    if method == "cot":
        return predict_cot(formula, disease, model)
    if method == "rag":
        return predict_rag(
            formula,
            disease,
            model,
            label=label,
            sample_id=sample_id,
            knowledge_cutoff_date=knowledge_cutoff_date,
            drug_names=drug_names,
            drug_identifiers=drug_identifiers,
            drug_inchikey=drug_inchikey,
        )
    raise ValueError(f"Unsupported method: {method}")


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def save_checkpoint(results, y_true, y_pred, processed_indices, checkpoint_path):
    checkpoint_data = {
        "results": results,
        "y_true": y_true,
        "y_pred": y_pred,
        "processed_indices": list(processed_indices),
        "timestamp": datetime.now().isoformat(),
        "total_processed": len(results),
    }
    try:
        with open(checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint_data, handle, indent=2, ensure_ascii=False)
        print(f"Checkpoint saved: {len(results)} samples processed")
    except Exception as exc:
        print(f"Error saving checkpoint: {exc}")


def load_checkpoint(checkpoint_path):
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                checkpoint_data = json.load(handle)
            print(f"Loaded checkpoint: {checkpoint_data['total_processed']} samples already processed")
            return (
                checkpoint_data["results"],
                checkpoint_data["y_true"],
                checkpoint_data["y_pred"],
                set(checkpoint_data["processed_indices"]),
                checkpoint_data["timestamp"],
            )
    except Exception as exc:
        print(f"Error loading checkpoint: {exc}")
    return [], [], [], set(), None


def get_checkpoint_path(args):
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    json_base = os.path.splitext(os.path.basename(args.json_path))[0]
    checkpoint_name = f"checkpoint_{json_base}_{args.method}_{args.backbone}_{args.agent_version}.json"
    return os.path.join(checkpoint_dir, checkpoint_name)


def cleanup_checkpoint(checkpoint_path):
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Checkpoint file removed: {checkpoint_path}")
    except Exception as exc:
        print(f"Warning: Could not remove checkpoint file: {exc}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run TreatAgent predictions.")
    parser.add_argument("--json_path", type=str, default="data/benchmark/split_inputs/drug_disjoint_test.json", help="Path to JSON file with disease and SMILES data")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for binary classification. Defaults to TREATAGENT_ARG_THRESHOLD/0.36 for eg/full and 0.5 for ls.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="multiagent",
        choices=["direct", "cot", "rag", "multiagent"],
        help="Prediction method to use",
    )
    parser.add_argument("--backbone", type=str, default="gpt-4o", help="Backbone model. Use --backbone local to disable LLM synthesis and run fully local multiagent inference.")
    parser.add_argument(
        "--agent_version",
        type=str,
        default="eg",
        choices=["eg", "full", "ls"],
        help="TreatAgent version for multiagent runs: eg=ArgumentGraph fallback scorer, full=LLM planner plus EvidenceGraph LLM judge, ls=LLM synthesis scorer baseline.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N samples")
    parser.add_argument("--generate_report", action="store_true", help="Generate an interactive HTML report for each sample")
    parser.add_argument("--use_memory", action="store_true", help="Enable vector-memory retrieval and storage")
    parser.add_argument(
        "--knowledge_cutoff_date",
        type=str,
        default=None,
        help="Optional knowledge cutoff date (YYYY-MM-DD). KB records with later snapshot_date will be ignored.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"model: {args.backbone}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    data = load_json_data(args.json_path)
    if not data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(data)} samples from {args.json_path}")
    print(f"Using method: {args.method}")
    if args.method == "multiagent":
        print(f"Using TreatAgent version: {args.agent_version}")

    checkpoint_path = get_checkpoint_path(args)
    y_true = []
    y_pred = []
    results = []
    processed_indices = set()

    if args.resume:
        results, y_true, y_pred, processed_indices, _ = load_checkpoint(checkpoint_path)
        if processed_indices:
            print(f"Resuming from previous run - {len(processed_indices)} samples already processed")

    total_samples = len(data)

    for idx, sample in enumerate(data):
        if idx in processed_indices:
            continue

        print(f"\nProcessing sample {idx + 1}/{total_samples}...")

        smiles = sample.get("smiles", "").strip()
        disease = sample.get("disease", "").strip()
        label = sample.get("label", 0)
        drug_names = sample.get("drugs") or sample.get("drug_names") or sample.get("drug") or []
        if isinstance(drug_names, str):
            drug_names = [drug_names]
        drug_identifiers = sample.get("drug_identifiers") or sample.get("identifiers") or []
        if isinstance(drug_identifiers, (str, int, float)):
            drug_identifiers = [drug_identifiers]
        drug_inchikey = sample.get("inchikey") or sample.get("drug_inchikey")

        try:
            sample_id = sample.get("sample_id") or build_sample_id(smiles, disease)
            prediction_output = predict_therapeutic_potential(
                smiles,
                disease,
                method=args.method,
                model=args.backbone,
                agent_version=args.agent_version,
                label=label,
                sample_id=sample_id,
                threshold=args.threshold,
                generate_report=args.generate_report,
                use_memory=args.use_memory,
                knowledge_cutoff_date=args.knowledge_cutoff_date,
                drug_names=drug_names,
                drug_identifiers=drug_identifiers,
                drug_inchikey=drug_inchikey,
            )

            rag_context = None
            rag_response = None
            if args.method == "multiagent":
                prediction_score = prediction_output["prediction_binary"]
                analysis = prediction_output["analysis"]
            elif args.method == "rag":
                prediction_score = prediction_output["prediction_binary"]
                analysis = prediction_output["analysis"]
                rag_context = prediction_output.get("rag_context")
                rag_response = prediction_output.get("rag_response")
            else:
                prediction_score = prediction_output
                analysis = None

            y_true.append(label)
            y_pred.append(prediction_score)

            result_entry = {
                "index": idx + 1,
                "sample_id": sample_id,
                "smiles": smiles,
                "disease": disease,
                "drug_names": drug_names,
                "drug_identifiers": drug_identifiers,
                "drug_inchikey": drug_inchikey,
                "label": label,
                "method": args.method,
                "agent_version": args.agent_version if args.method == "multiagent" else None,
                "prediction_score": int(prediction_score),
                "prediction_binary": int(prediction_score),
            }

            if analysis is not None:
                result_entry.update(
                    {
                        "sample_id": analysis.get("sample_id", sample_id),
                        "agent_version": analysis.get("agent_version", args.agent_version),
                        "llm_planner_enabled": analysis.get("llm_planner_enabled"),
                        "llm_explanation_enabled": analysis.get("llm_explanation_enabled"),
                        "llm_judge_enabled": analysis.get("llm_judge_enabled"),
                        "llm_synthesis_enabled": analysis.get("llm_synthesis_enabled"),
                        "llm_experts_enabled": analysis.get("llm_experts_enabled"),
                        "llm_expert_names": analysis.get("llm_expert_names"),
                        "force_all_experts": analysis.get("force_all_experts"),
                        "derived_argument_claims_enabled": analysis.get("derived_argument_claims_enabled"),
                        "disabled_experts": analysis.get("disabled_experts"),
                        "raw_score": analysis["synthesis"]["raw_score"],
                        "synthesis_source": analysis["synthesis"].get("synthesis_source"),
                        "explanation_source": analysis["synthesis"].get("explanation_source"),
                        "group_scores": analysis["synthesis"].get("group_scores"),
                        "calibrated_probability": analysis["calibration"]["calibrated_probability"],
                        "argument_probability": analysis.get("argument_probability"),
                        "argument_prediction": analysis.get("argument_prediction"),
                        "argument_factors": analysis.get("argument_factors"),
                        "llm_judge": analysis.get("llm_judge"),
                        "llm_judge_probability": analysis.get("llm_judge_probability"),
                        "llm_judge_prediction": analysis.get("llm_judge_prediction"),
                        "final_score_source": analysis.get("final_score_source"),
                        "final_threshold": analysis.get("final_threshold"),
                        "synthesis_explanation": analysis["synthesis"]["explanation"],
                        "trajectory": analysis["trajectory"],
                        "evidence_graph": analysis["evidence_graph"],
                        "expert_outputs": analysis["expert_outputs"],
                        "report_path": analysis["report_path"],
                        "report_summary": analysis["report_summary"],
                        "memory_similar_cases": analysis["memory_similar_cases"],
                        "memory_enabled": analysis["memory_enabled"],
                        "memory_init_error": analysis["memory_init_error"],
                        "stored_case_id": analysis["stored_case_id"],
                        "knowledge_cutoff_date": analysis.get("knowledge_cutoff_date"),
                    }
                )
            if args.method == "rag":
                result_entry.update(
                    {
                        "rag_context": rag_context,
                        "rag_response": rag_response,
                    }
                )
            results.append(result_entry)
            processed_indices.add(idx)

            print(f"  SMILES: {smiles[:50]}...")
            print(f"  Sample ID: {sample_id}")
            print(f"  Disease: {disease}")
            print(f"  Label: {label}, Predicted: {int(prediction_score)} (Method: {args.method})")
            if analysis is not None:
                print(f"  Raw score: {analysis['synthesis']['raw_score']:.4f}/10")
                print(f"  Synthesis source: {analysis['synthesis'].get('synthesis_source')}")
                print(f"  Explanation source: {analysis['synthesis'].get('explanation_source')}")
                print(f"  Calibrated probability: {analysis['calibration']['calibrated_probability']:.4f}")
                if analysis.get("argument_probability") is not None:
                    print(f"  ARG probability: {analysis['argument_probability']:.4f}")
                if analysis.get("llm_judge_probability") is not None:
                    print(f"  LLM judge probability: {analysis['llm_judge_probability']:.4f}")
                print(f"  Final score source: {analysis.get('final_score_source')}")
                print(f"  Group scores: {analysis['synthesis'].get('group_scores')}")
                print(f"  Similar memory cases: {analysis['memory_similar_cases']}")
                print(f"  Memory enabled: {analysis['memory_enabled']}")
                if analysis["memory_init_error"]:
                    print(f"  Memory init error: {analysis['memory_init_error']}")
                if analysis["report_path"]:
                    print(f"  Report: {analysis['report_path']}")

            if (idx + 1) % args.save_every == 0:
                save_checkpoint(results, y_true, y_pred, processed_indices, checkpoint_path)

        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user. Saving checkpoint...")
            save_checkpoint(results, y_true, y_pred, processed_indices, checkpoint_path)
            print("Checkpoint saved. You can resume later with --resume flag.")
            return
        except Exception as exc:
            print(f"Error processing sample {idx + 1}: {repr(exc)}")
            traceback.print_exc()
            continue

    if y_true and y_pred:
        metrics = compute_metrics(y_true, y_pred)
        effective_threshold = args.threshold
        if effective_threshold is None:
            if args.method == "multiagent" and args.agent_version in {"eg", "full"}:
                effective_threshold = _resolve_arg_threshold(None)
            else:
                effective_threshold = 0.5

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Method: {args.method}")
        if args.method == "multiagent":
            print(f"TreatAgent version: {args.agent_version}")
        print(f"Total samples processed: {len(y_true)}")
        print(f"Positive samples: {sum(y_true)}")
        print(f"Negative samples: {len(y_true) - sum(y_true)}")
        print(f"Threshold: {effective_threshold}")
        print()
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print("=" * 50)

        results_dir = f"results/{args.backbone}"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_suffix = f"_{args.agent_version}" if args.method == "multiagent" else ""
        output_file = f"{results_dir}/results_{args.method}{version_suffix}_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metrics": metrics,
                    "method": args.method,
                    "agent_version": args.agent_version if args.method == "multiagent" else None,
                    "threshold": effective_threshold,
                    "total_samples": len(y_true),
                    "positive_samples": sum(y_true),
                    "negative_samples": len(y_true) - sum(y_true),
                    "results": results,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nDetailed results saved to: {output_file}")
        cleanup_checkpoint(checkpoint_path)
    else:
        print("No samples were processed successfully.")

    print(f"\nProcessing completed. Total samples processed: {len(results)}")


if __name__ == "__main__":
    main()
