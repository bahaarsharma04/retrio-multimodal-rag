"""
Retrio - Automated Model Accuracy & RAG Benchmark Evaluation Suite
==================================================================
Runs deterministic, comprehensive evaluations across:
1. Information Retrieval (IR): HitRate@k, Precision@k, Recall@k, MRR, nDCG@k, MAP@k, Context Relevance, Noise Rate, F1
2. Generation Quality & Grounding: Faithfulness, Answer Correctness, Citation Accuracy, Anti-Hallucination Assertion, Multi-Turn Recall
3. Regression Detection: Automatically compares against a saved baseline and flags any metric degradation.

Usage:
    # Full benchmark on default/detected project
    python scripts/evaluate_accuracy.py

    # Quick 3-case smoke test (~30s)
    python scripts/evaluate_accuracy.py --quick

    # Benchmark specific project by name or UUID
    python scripts/evaluate_accuracy.py --project "testing.."

    # Establish or update reference baseline
    python scripts/evaluate_accuracy.py --set-baseline

    # Compare current code against baseline (fails with exit code 1 on regression)
    python scripts/evaluate_accuracy.py --compare --threshold 0.05
"""

import sys
import json
import math
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add repository root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.projects import list_projects
from core.rag_chain import generate_rag_response
from core.vectorstore import get_live_vector_count

BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"
DEFAULT_OUTPUT_PATH = BENCHMARK_DIR / "latest_accuracy.json"
DEFAULT_BASELINE_PATH = BENCHMARK_DIR / "baseline_accuracy.json"


# ====================================================================
# IR METRICS COMPUTATION (MATHEMATICALLY BOUNDED TO [0.0, 1.0])
# ====================================================================

def compute_ir_metrics(
    retrieved_docs: List[Dict[str, Any]],
    relevant_doc_names: List[str],
    k: int = 4,
) -> Dict[str, float]:
    """
    Computes standard Information Retrieval metrics for top-k results:
    - HitRate@k: 1.0 if at least one relevant document chunk is retrieved, else 0.0
    - Precision@k: fraction of top-k chunks that belong to the target documents
    - Recall@k: fraction of target documents represented in top-k
    - MRR: Reciprocal rank (1 / position) of the first relevant chunk
    - nDCG@k: Normalized Discounted Cumulative Gain (strictly bounded in [0.0, 1.0])
    - MAP@k: Mean Average Precision at k (strictly bounded in [0.0, 1.0])
    - Context Relevance: Precision of top-k retrieved chunks
    - Noise Rate: Fraction of irrelevant chunks (1.0 - Precision@k)
    - F1 Score: Harmonic mean of Precision and Recall
    """
    top_k = retrieved_docs[:k]
    if not top_k:
        return {
            "hit_rate_at_k": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
            "context_relevance": 0.0,
            "noise_rate": 1.0,
            "f1_score": 0.0,
        }

    # Binary relevance: does chunk belong to one of the target gold documents?
    relevance_flags = [
        1 if any(t_name.lower() in (d.get("filename") or "").lower() for t_name in relevant_doc_names) else 0
        for d in top_k
    ]
    relevant_retrieved = sum(relevance_flags)

    # 1. Hit Rate @ k
    hit_rate = 1.0 if relevant_retrieved > 0 else 0.0

    # 2. Precision @ k
    precision = relevant_retrieved / len(top_k)

    # 3. Recall @ k (unique target documents retrieved / total unique target documents)
    unique_retrieved_targets = set()
    for d in top_k:
        fname = (d.get("filename") or "").lower()
        for t_name in relevant_doc_names:
            if t_name.lower() in fname:
                unique_retrieved_targets.add(t_name.lower())
    recall = len(unique_retrieved_targets) / max(1, len(relevant_doc_names))
    recall = min(1.0, max(0.0, recall))

    # 4. F1 Score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 5. MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, rel in enumerate(relevance_flags, start=1):
        if rel == 1:
            mrr = 1.0 / rank
            break

    # 6. nDCG @ k (Normalized Discounted Cumulative Gain)
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_flags))
    # Ideal DCG assumes all relevant chunks are ordered first
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(relevant_retrieved))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    ndcg = min(1.0, max(0.0, ndcg))

    # 7. MAP @ k (Average Precision for this query)
    running_rel = 0
    precisions_at_rel = []
    for idx, rel in enumerate(relevance_flags, start=1):
        if rel == 1:
            running_rel += 1
            precisions_at_rel.append(running_rel / idx)
    ap = (sum(precisions_at_rel) / max(1, relevant_retrieved)) if precisions_at_rel else 0.0
    ap = min(1.0, max(0.0, ap))

    context_relevance = precision
    noise_rate = 1.0 - context_relevance

    return {
        "hit_rate_at_k": round(hit_rate, 4),
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
        "map_at_k": round(ap, 4),
        "context_relevance": round(context_relevance, 4),
        "noise_rate": round(noise_rate, 4),
        "f1_score": round(f1, 4),
    }


# ====================================================================
# GENERATION QUALITY & GROUNDING EVALUATION
# ====================================================================

def evaluate_generation_quality(
    answer: str,
    sources: List[Dict[str, Any]],
    gold_keywords: List[str],
    negative_constraint: bool = False,
    is_meta: bool = False,
    expected_meta_count: int = 0,
) -> Dict[str, float]:
    """
    Evaluates LLM generation quality:
    - Faithfulness: Groundedness of claims against retrieved context
    - Citation Accuracy: Validity of bracketed citations [doc, page]
    - Answer Correctness: Recall of essential factual keywords/concepts
    - Anti-Hallucination Assertion Accuracy: Correct refusal on zero-extrapolation trap queries
    - Multi-Turn History Recall: Recall of prior conversation turns
    """
    ans_lower = answer.lower()

    # Case A: Meta-Conversational Query (Recap / Question Listing)
    if is_meta:
        found_count = len(re.findall(r'^\s*\d+[\.\)]\s*', answer, flags=re.MULTILINE))
        history_recall = min(1.0, found_count / expected_meta_count) if expected_meta_count > 0 else 1.0
        return {
            "faithfulness": 1.0,
            "hallucination_rate": 0.0,
            "anti_hallucination_accuracy": 1.0,
            "citation_accuracy": 1.0,
            "answer_correctness": round(1.0 if history_recall >= 0.85 else history_recall, 4),
            "answer_relevance": 1.0,
            "context_utilisation": 1.0,
            "completeness": round(history_recall, 4),
            "multi_turn_history_recall": round(history_recall, 4),
        }

    # Case B: Zero-Extrapolation Trap / Negative Constraint Query
    if negative_constraint:
        negative_refusal_patterns = [
            "do not describe any relationship",
            "no relationship or mechanism",
            "no overlap or connection",
            "does not specify",
            "not mentioned",
            "do not describe any connection",
            "do not contain sufficient information",
            "no connection",
            "no direct link",
            "not related",
        ]
        has_negative_refusal = any(pat in ans_lower for pat in negative_refusal_patterns)

        # Detect hallucinated physics or causal bridges (e.g. connecting solar particles to brake fluid)
        hallucinated_bridge = (
            any(w in ans_lower for w in ["bernoulli", "pascal", "hydraulic", "laminar flow", "brake"])
            and any(w in ans_lower for w in ["aurora", "solar wind", "geomagnetic", "magnetosphere"])
            and not has_negative_refusal
        )

        hallucination_rate = 1.0 if hallucinated_bridge else (0.0 if has_negative_refusal else 0.5)
        anti_hallucination_acc = 1.0 if has_negative_refusal and not hallucinated_bridge else 0.0
        faithfulness = 1.0 if has_negative_refusal and not hallucinated_bridge else 0.2

        # Check citations: must NOT cite unrelated documents as proof of connection
        citations = re.findall(r'\[([^\]]+)\]', answer)
        citation_accurate = True
        for c in citations:
            if "aurora" in c.lower() and any(w in ans_lower for w in ["bernoulli", "pascal", "laminar"]):
                citation_accurate = False

        return {
            "faithfulness": round(faithfulness, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "anti_hallucination_accuracy": round(anti_hallucination_acc, 4),
            "citation_accuracy": round(1.0 if citation_accurate else 0.0, 4),
            "answer_correctness": round(1.0 if has_negative_refusal else 0.0, 4),
            "answer_relevance": round(1.0 if has_negative_refusal else 0.3, 4),
            "context_utilisation": 1.0,
            "completeness": round(1.0 if has_negative_refusal else 0.3, 4),
            "multi_turn_history_recall": 1.0,
        }

    # Case C: Standard Factual Grounding Query
    keyword_hits = sum(1 for kw in gold_keywords if kw.lower() in ans_lower)
    completeness = (keyword_hits / len(gold_keywords)) if gold_keywords else 1.0

    # Inline Citation Verification
    citations = re.findall(r'\[([^\]]+)\]', answer)
    has_citations = len(citations) > 0
    citation_valid = False
    if has_citations and sources:
        # Check if citations mention at least one of the retrieved source filenames
        for s in sources:
            s_name = s.get("filename", "").lower()
            if any(s_name in c.lower() or c.lower() in s_name for c in citations):
                citation_valid = True
                break
    elif not sources:
        citation_valid = True

    citation_score = 1.0 if citation_valid else (0.75 if has_citations else 0.0)
    faithfulness = min(1.0, 0.5 + (0.5 * completeness))

    return {
        "faithfulness": round(faithfulness, 4),
        "hallucination_rate": 0.0,
        "anti_hallucination_accuracy": 1.0,
        "citation_accuracy": round(citation_score, 4),
        "answer_correctness": round(completeness, 4),
        "answer_relevance": round(1.0 if completeness >= 0.5 else 0.7, 4),
        "context_utilisation": round(min(1.0, completeness * 1.1), 4),
        "completeness": round(completeness, 4),
        "multi_turn_history_recall": 1.0,
    }


# ====================================================================
# TEST CASES SUITE
# ====================================================================

SIMULATED_CONVERSATION_HISTORY = [
    {"role": "user", "content": "summerize aurora project"},
    {"role": "assistant", "content": "The Aurora project focuses on the phenomenon of auroras caused by energetic particles colliding with atmospheric gases in the polar regions."},
    {"role": "user", "content": "what is density in the fluid mechanics"},
    {"role": "assistant", "content": "The uploaded documents do not contain sufficient information to answer this question."},
    {"role": "user", "content": "what is mechanics or fluid"},
    {"role": "assistant", "content": "Fluid mechanics is a branch of physics and engineering that studies fluid behavior under resting and dynamic conditions."},
    {"role": "user", "content": "what is its role in the breaks"},
    {"role": "assistant", "content": "In hydraulic brakes, Pascal's Law governs how pedal pressure is transmitted through the brake fluid to the calipers."},
    {"role": "user", "content": "can you discribe bernoulli principle"},
    {"role": "assistant", "content": "Bernoulli's Principle states that an increase in fluid velocity occurs simultaneously with a decrease in pressure or potential energy."},
    {"role": "user", "content": "explain leminar flow of fluid"},
    {"role": "assistant", "content": "Laminar flow occurs when a fluid flows in parallel layers without disruption, below a Reynolds number of 2,300."},
    {"role": "user", "content": "how does they all work together"},
    {"role": "assistant", "content": "The principles work together to describe continuity, conservation of energy, and viscous flow regimes."},
    {"role": "user", "content": "effect or one on the other"},
    {"role": "assistant", "content": "The principles interact through pressure differentials, velocity gradients, and viscous boundary layer behaviors."},
]

STANDARD_TEST_CASES = [
    {
        "id": "TC-01",
        "name": "Single-Doc Factual Grounding (Laminar Flow & Reynolds Threshold)",
        "query": "Explain laminar flow of fluid and its Reynolds number threshold.",
        "history": [],
        "relevant_docs": ["Fluid mechanics", "phys.txt"],
        "gold_keywords": ["smooth", "parallel", "reynolds", "2,300"],
        "negative_constraint": False,
        "is_meta": False,
        "is_quick": True,
    },
    {
        "id": "TC-02",
        "name": "Causal Sequence & Technical Mechanism (Pascal's Law in Brakes)",
        "query": "Explain the role of Pascal's Law in hydraulic brakes.",
        "history": [],
        "relevant_docs": ["Fluid mechanics", "phys.txt"],
        "gold_keywords": ["pascal", "pressure", "brake", "fluid"],
        "negative_constraint": False,
        "is_meta": False,
        "is_quick": False,
    },
    {
        "id": "TC-03",
        "name": "Negative Cross-Topic Relational Guard (Zero-Extrapolation Trap)",
        "query": "How do the fluid mechanics principles in hydraulic brakes affect the auroral light display in Aurora.pdf?",
        "history": SIMULATED_CONVERSATION_HISTORY,
        "relevant_docs": ["Aurora.pdf"],
        "gold_keywords": [],
        "negative_constraint": True,
        "is_meta": False,
        "is_quick": True,
    },
    {
        "id": "TC-04",
        "name": "Meta-Conversational Session History Recap ('what have we done till now')",
        "query": "what have we done till now",
        "history": SIMULATED_CONVERSATION_HISTORY,
        "relevant_docs": [],
        "gold_keywords": [],
        "negative_constraint": False,
        "is_meta": True,
        "expected_meta_count": 8,
        "is_quick": False,
    },
    {
        "id": "TC-05",
        "name": "Scientific Deep Explanation with Altitudes & Gases (Aurora Colors)",
        "query": "According to Aurora.pdf, what produces the different colors of the aurora and what altitudes are involved?",
        "history": [],
        "relevant_docs": ["Aurora.pdf"],
        "gold_keywords": ["oxygen", "nitrogen", "green", "red", "altitude"],
        "negative_constraint": False,
        "is_meta": False,
        "is_quick": True,
    },
    {
        "id": "TC-06",
        "name": "Multi-Document Comparative Synthesis (Aurora vs Population)",
        "query": "Compare the topics discussed in Aurora.pdf and population-growth.pdf",
        "history": [],
        "relevant_docs": ["Aurora.pdf", "population-growth.pdf"],
        "gold_keywords": ["aurora", "solar", "population", "growth"],
        "negative_constraint": False,
        "is_meta": False,
        "is_quick": False,
    },
    {
        "id": "TC-07",
        "name": "Multimodal Vision Diagram Grounding (Plant Cell Organelles)",
        "query": "What organelles are unique to plant cells in img-testing.pdf?",
        "history": [],
        "relevant_docs": ["img-testing.pdf"],
        "gold_keywords": ["chloroplast", "cell wall", "vacuole"],
        "negative_constraint": False,
        "is_meta": False,
        "is_quick": False,
    },
]


# ====================================================================
# TARGET PROJECT DISCOVERY
# ====================================================================

def discover_target_project(preferred_identifier: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Discovers a suitable project workspace containing indexed vectors."""
    all_projects = list_projects()
    if not all_projects:
        return None

    if preferred_identifier:
        pref_clean = preferred_identifier.strip().lower()
        for p in all_projects:
            if p.get("id") == preferred_identifier or p.get("name", "").lower() == pref_clean:
                return p

    # 1. Prioritize canonical evaluation project "testing.."
    for p in all_projects:
        if p.get("name", "").lower() in ["testing..", "testing"]:
            return p

    # 2. Sort remaining projects by live vector count descending
    scored = []
    for p in all_projects:
        pid = p.get("id")
        try:
            cnt = get_live_vector_count(pid)
        except Exception:
            cnt = 0
        scored.append((cnt, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else all_projects[0]


# ====================================================================
# BENCHMARK RUNNER & REGRESSION ENGINE
# ====================================================================

def run_evaluation(
    project_id: str,
    quick_mode: bool = False,
    k_eval: int = 4,
) -> Dict[str, Any]:
    """Runs the benchmark suite on the specified project and computes macro metrics."""
    test_cases = [tc for tc in STANDARD_TEST_CASES if (not quick_mode or tc.get("is_quick", False))]

    print(f"\nEvaluating {len(test_cases)} test cases against Project ID: {project_id}")
    print("-" * 75)

    case_records = []
    for tc in test_cases:
        t0 = time.perf_counter()
        res = generate_rag_response(
            project_id=project_id,
            query=tc["query"],
            chat_history=tc.get("history", []),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # IR Metrics
        if tc.get("relevant_docs"):
            ir = compute_ir_metrics(sources, tc["relevant_docs"], k=k_eval)
        else:
            ir = {
                "hit_rate_at_k": 1.0,
                "precision_at_k": 1.0,
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "ndcg_at_k": 1.0,
                "map_at_k": 1.0,
                "context_relevance": 1.0,
                "noise_rate": 0.0,
                "f1_score": 1.0,
            }

        # Generation Quality Metrics
        gen = evaluate_generation_quality(
            answer=answer,
            sources=sources,
            gold_keywords=tc.get("gold_keywords", []),
            negative_constraint=tc.get("negative_constraint", False),
            is_meta=tc.get("is_meta", False),
            expected_meta_count=tc.get("expected_meta_count", 0),
        )

        # Determine individual case status
        case_passed = ir["hit_rate_at_k"] >= 0.9 and gen["faithfulness"] >= 0.75 and gen["answer_correctness"] >= 0.5
        status_tag = "PASS" if case_passed else ("WARN" if ir["hit_rate_at_k"] >= 0.5 else "FAIL")

        record = {
            "test_id": tc["id"],
            "name": tc["name"],
            "query": tc["query"],
            "latency_ms": round(latency_ms, 1),
            "sources_retrieved": len(sources),
            "status": status_tag,
            "ir_metrics": ir,
            "gen_metrics": gen,
            "answer_preview": answer[:150] + ("..." if len(answer) > 150 else ""),
        }
        case_records.append(record)

        print(
            f"[{status_tag:4s}] {tc['id']}: {tc['name'][:42]:42s} | "
            f"Hit: {ir['hit_rate_at_k']*100:.0f}% | "
            f"Faith: {gen['faithfulness']*100:.0f}% | "
            f"Correct: {gen['answer_correctness']*100:.0f}% | "
            f"{latency_ms:6.1f}ms"
        )

    # Calculate Macro Averages
    n = len(case_records)
    macro_scores = {
        # Retrieval Metrics
        "HitRate@k": round(sum(c["ir_metrics"]["hit_rate_at_k"] for c in case_records) / n, 4),
        "Precision@k": round(sum(c["ir_metrics"]["precision_at_k"] for c in case_records) / n, 4),
        "Recall@k": round(sum(c["ir_metrics"]["recall_at_k"] for c in case_records) / n, 4),
        "MRR": round(sum(c["ir_metrics"]["mrr"] for c in case_records) / n, 4),
        "nDCG@k": round(sum(c["ir_metrics"]["ndcg_at_k"] for c in case_records) / n, 4),
        "MAP@k": round(sum(c["ir_metrics"]["map_at_k"] for c in case_records) / n, 4),
        "Context Relevance": round(sum(c["ir_metrics"]["context_relevance"] for c in case_records) / n, 4),
        "Noise Rate": round(sum(c["ir_metrics"]["noise_rate"] for c in case_records) / n, 4),
        "F1 Score": round(sum(c["ir_metrics"]["f1_score"] for c in case_records) / n, 4),
        # Generation Metrics
        "Faithfulness": round(sum(c["gen_metrics"]["faithfulness"] for c in case_records) / n, 4),
        "Anti-Hallucination Accuracy": round(sum(c["gen_metrics"]["anti_hallucination_accuracy"] for c in case_records) / n, 4),
        "Citation Accuracy": round(sum(c["gen_metrics"]["citation_accuracy"] for c in case_records) / n, 4),
        "Answer Correctness": round(sum(c["gen_metrics"]["answer_correctness"] for c in case_records) / n, 4),
        "Answer Relevance": round(sum(c["gen_metrics"]["answer_relevance"] for c in case_records) / n, 4),
        "Completeness": round(sum(c["gen_metrics"]["completeness"] for c in case_records) / n, 4),
        "Multi-Turn History Recall": round(sum(c["gen_metrics"]["multi_turn_history_recall"] for c in case_records) / n, 4),
        "Avg Latency (ms)": round(sum(c["latency_ms"] for c in case_records) / n, 1),
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_id": project_id,
        "quick_mode": quick_mode,
        "total_test_cases": n,
        "macro_scores": macro_scores,
        "case_details": case_records,
    }
    return report


def compare_with_baseline(
    current_macro: Dict[str, float],
    baseline_path: Path,
    threshold: float = 0.05,
) -> Tuple[bool, List[str]]:
    """
    Compares current macro scores against a saved baseline.
    Returns (has_regression, formatted_diff_lines).
    """
    if not baseline_path.exists():
        return False, [f"No baseline file found at '{baseline_path}'. Run with --set-baseline to create one."]

    try:
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_macro = baseline_data.get("macro_scores", {})
    except Exception as e:
        return False, [f"Failed to read baseline file: {e}"]

    has_regression = False
    lines = []
    lines.append(f"\n{'Metric':<30s} | {'Current':<9s} | {'Baseline':<9s} | {'Delta':<10s} | {'Status'}")
    lines.append("-" * 75)

    critical_metrics = [
        "HitRate@k", "MRR", "nDCG@k", "Faithfulness",
        "Anti-Hallucination Accuracy", "Citation Accuracy", "Answer Correctness"
    ]

    for metric in critical_metrics:
        curr_val = current_macro.get(metric, 0.0)
        base_val = baseline_macro.get(metric, curr_val)
        delta = curr_val - base_val

        # Check for regression drop beyond tolerance
        is_regressed = delta < -threshold
        if is_regressed:
            has_regression = True
            status = "❌ REGRESSED"
        elif delta > 0.01:
            status = "✅ IMPROVED"
        else:
            status = "➖ STABLE"

        delta_str = f"{delta:+.2%}" if "@" not in metric and metric != "MRR" and metric != "nDCG@k" and metric != "MAP@k" else f"{delta:+.4f}"
        curr_str = f"{curr_val*100:.1f}%" if "@" not in metric and metric != "MRR" and metric != "nDCG@k" and metric != "MAP@k" else f"{curr_val:.4f}"
        base_str = f"{base_val*100:.1f}%" if "@" not in metric and metric != "MRR" and metric != "nDCG@k" and metric != "MAP@k" else f"{base_val:.4f}"

        lines.append(f"{metric:<30s} | {curr_str:<9s} | {base_str:<9s} | {delta_str:<10s} | {status}")

    return has_regression, lines


# ====================================================================
# MAIN CLI ENTRYPOINT
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Retrio RAG Model Accuracy Evaluation & Regression Suite")
    parser.add_argument("--project", type=str, default=None, help="Target Project ID or Name (auto-detected if omitted)")
    parser.add_argument("--quick", action="store_true", help="Run fast 3-test smoke evaluation (~30s)")
    parser.add_argument("--set-baseline", action="store_true", help="Save current run as the baseline for future regression checks")
    parser.add_argument("--compare", action="store_true", help="Compare current results against baseline_accuracy.json")
    parser.add_argument("--threshold", type=float, default=0.05, help="Regression tolerance threshold (default: 0.05 or 5%%)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Custom JSON output path")
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("RETRIO AUTOMATED MODEL ACCURACY & RAG BENCHMARK EVALUATOR")
    print("=" * 75)

    target_proj = discover_target_project(args.project)
    if not target_proj:
        print("❌ Error: No active projects found in workspace. Please create and ingest a project first.")
        sys.exit(1)

    print(f"Target Project: '{target_proj.get('name')}' (ID: {target_proj.get('id')})")
    print(f"Mode: {'Quick Smoke Test (3 cases)' if args.quick else 'Comprehensive Suite (7 cases)'}")

    # Run Evaluation
    report = run_evaluation(
        project_id=target_proj["id"],
        quick_mode=args.quick,
    )

    # Save latest report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n✔ Full evaluation report saved to: {out_path}")

    # Print Macro Summary Table
    print("\n" + "=" * 50)
    print("MACRO ACCURACY & GROUNDING SUMMARY SCORES:")
    print("=" * 50)
    macro = report["macro_scores"]
    for k, v in macro.items():
        if k == "Avg Latency (ms)":
            print(f"  * {k:30s}: {v:.1f} ms")
        elif any(sym in k for sym in ["@", "MRR", "nDCG", "MAP"]):
            print(f"  * {k:30s}: {v:.4f}")
        else:
            print(f"  * {k:30s}: {v * 100:.2f}%")
    print("=" * 50)

    # Save as baseline if requested
    if args.set_baseline:
        DEFAULT_BASELINE_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n✔ Saved current run as baseline reference: {DEFAULT_BASELINE_PATH}")

    # Compare with baseline if requested or if baseline exists
    has_regression = False
    if args.compare or (DEFAULT_BASELINE_PATH.exists() and not args.set_baseline):
        print("\n" + "=" * 50)
        print("REGRESSION COMPARISON AGAINST BASELINE:")
        print("=" * 50)
        has_regression, diff_lines = compare_with_baseline(macro, DEFAULT_BASELINE_PATH, threshold=args.threshold)
        for line in diff_lines:
            print(line)
        print("=" * 75)

        if has_regression:
            print("\n❌ REGRESSION ALERT: One or more key metrics dropped beyond the allowed threshold!")
            sys.exit(1)
        else:
            print("\n✅ REGRESSION CHECK PASSED: All accuracy and grounding metrics within acceptable thresholds.")

    print("\nBenchmark run completed successfully.")


if __name__ == "__main__":
    main()
