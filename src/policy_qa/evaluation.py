import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI, OpenAI, RateLimitError
from ragas.dataset_schema import SingleTurnSample

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from .pipeline import RAGPipeline
from config.settings import SETTINGS


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


# =============================================================
# Load JSONL evaluation dataset
# =============================================================
def load_jsonl(path: Path):
    rows = []

    with open(
        path,
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


async def score_metric(metric, sample, fields):
    """Score a collection metric using fields from a RAGAS sample."""
    values = sample.model_dump(
        include=set(fields),
        exclude_none=True,
    )
    return (await metric.ascore(**values)).value


def normalize_document_name(document):
    """Match CUAD names with indexed PDF names."""
    value = str(document).strip()
    if value.casefold().endswith(".pdf"):
        value = value[:-4]
    return value.casefold()


def format_score(value):
    return f"{value:.4f}" if value is not None else "N/A"


def row_identifier(row):
    """Return a stable identifier shared by the dataset and checkpoint."""
    qa_id = row.get("qa_id")
    if qa_id:
        return f"qa_id:{qa_id}"
    return row_fallback_identifier(row)


def row_fallback_identifier(row):
    return f"question:{row.get('question', '')}\ndocument:{row.get('document', '')}"


def row_match_keys(row):
    keys = {row_identifier(row), row_fallback_identifier(row)}
    if row.get("qa_id"):
        keys.add(f"qa_id:{row['qa_id']}")
    return keys


def is_rate_limit_error(error):
    """Detect direct and Instructor-wrapped OpenAI/Groq 429 errors."""
    pending = [error]
    seen = set()

    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, RateLimitError):
            return True

        if getattr(current, "status_code", None) == 429:
            return True

        for attribute in ("__cause__", "__context__", "last_exception"):
            nested = getattr(current, attribute, None)
            if isinstance(nested, BaseException):
                pending.append(nested)

        for attempt in getattr(current, "failed_attempts", ()) or ():
            nested = getattr(attempt, "exception", None)
            if isinstance(nested, BaseException):
                pending.append(nested)

        for argument in getattr(current, "args", ()):
            if isinstance(argument, BaseException):
                pending.append(argument)

    return False


def summarize(outputs):
    summary = {}
    for metric in METRIC_NAMES:
        values = [
            row.get("scores", {}).get(metric)
            for row in outputs
            if row.get("scores", {}).get(metric) is not None
        ]
        summary[metric] = sum(values) / len(values) if values else None
    return summary


def save_checkpoint(path, outputs):
    """Atomically persist completed rows and their current aggregates."""
    payload = {
        "summary": summarize(outputs),
        "num_questions": len(outputs),
        "rows": outputs,
    }
    temporary_path = path.with_suffix(".json.tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    return payload


def load_checkpoint(path):
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as file:
        payload = json.load(file)

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid checkpoint format: rows must be a list in {path}")
    return rows


# =============================================================
# Score evaluation rows
# =============================================================
async def score_rows(
    rows,
    pipeline,
    outputs,
    checkpoint_path,
):
    # ---------------------------------------------------------
    # RAGAS evaluator LLM
    #
    # Groq provides an OpenAI-compatible API.
    # Using the OpenAI client avoids the RAGAS/Groq SDK
    # Instructor adapter incompatibility.
    # ---------------------------------------------------------
    evaluator_client = OpenAI(
        api_key=SETTINGS.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # RAGAS collection metrics call agenerate(). Keep the working
    # OpenAI-compatible Groq client above and give RAGAS its async twin.
    evaluator_async_client = AsyncOpenAI(
        api_key=SETTINGS.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=120.0,
    )

    evaluator_llm = llm_factory(
        SETTINGS.llm_model,
        provider="openai",
        client=evaluator_async_client,
        temperature=0.0,
        max_tokens=4096,
    )

    # ---------------------------------------------------------
    # Evaluation embeddings
    # ---------------------------------------------------------
    evaluator_embeddings = HuggingFaceEmbeddings(
        model=SETTINGS.embedding_model,
        device="cpu",
        normalize_embeddings=True,
    )

    # ---------------------------------------------------------
    # RAGAS metrics
    # ---------------------------------------------------------
    metrics = [
        Faithfulness(
            llm=evaluator_llm
        ),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        ContextPrecision(
            llm=evaluator_llm
        ),
        ContextRecall(
            llm=evaluator_llm
        ),
    ]

    completed_ids = {
        key
        for row in outputs
        for key in row_match_keys(row)
    }
    skipped = 0
    interrupted = False

    # =========================================================
    # Evaluate each question
    # =========================================================
    for question_number, row in enumerate(
        rows,
        start=1,
    ):
        question = row["question"]

        document = row.get(
            "document",
            "",
        )

        identifiers = row_match_keys(row)
        if identifiers & completed_ids:
            skipped += 1
            print(
                f"[{question_number}/{len(rows)}] Skipping completed question."
            )
            continue

        reference = row.get(
            "ground_truth",
            "",
        )

        print(
            f"\n[{question_number}/{len(rows)}] "
            f"Evaluating..."
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Target document: {document}"
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # The target CUAD document is passed directly into
        # the retrieval layer.
        #
        # Dense retrieval + BM25 + RRF + reranking are
        # therefore restricted to this contract.
        # -----------------------------------------------------
        try:
            result = pipeline.answer(
                question,
                document=document,
            )

            # -----------------------------------------------------
            # Retrieved contexts. Keep the document constraint in
            # force even if a pipeline implementation returns a
            # malformed source.
            # -----------------------------------------------------
            target_document = normalize_document_name(document)
            sources = [
                source
                for source in result.get("sources", [])
                if normalize_document_name(source.get("document", ""))
                == target_document
            ]

            if len(sources) != len(result.get("sources", [])):
                print(
                    "Warning: discarded sources outside the target document."
                )

            contexts = [source["text"] for source in sources if source.get("text")]

            sample = SingleTurnSample(
                user_input=question,
                response=result.get("answer", ""),
                retrieved_contexts=contexts,
                reference=reference,
            )

            # -----------------------------------------------------
            # RAGAS scores
            # -----------------------------------------------------
            scores = {}

        # -----------------------------------------------------
        # 1. Faithfulness
        # -----------------------------------------------------
            scores["faithfulness"] = None
            if contexts:
                scores["faithfulness"] = await score_metric(
                    metrics[0],
                    sample,
                    ("user_input", "response", "retrieved_contexts"),
                )

        # -----------------------------------------------------
        # 2. Answer Relevancy
        # -----------------------------------------------------
            scores["answer_relevancy"] = await score_metric(
                metrics[1],
                sample,
                ("user_input", "response"),
            )

        # -----------------------------------------------------
        # 3. Context Precision
        # -----------------------------------------------------
            scores["context_precision"] = None
            if contexts and reference:
                scores["context_precision"] = await score_metric(
                    metrics[2],
                    sample,
                    ("user_input", "retrieved_contexts", "reference"),
                )

        # -----------------------------------------------------
        # 4. Context Recall
        # -----------------------------------------------------
            scores["context_recall"] = None
            if contexts and reference:
                scores["context_recall"] = await score_metric(
                    metrics[3],
                    sample,
                    ("user_input", "retrieved_contexts", "reference"),
                )

            # -----------------------------------------------------
            # Save result immediately after all four metrics finish.
            # -----------------------------------------------------
            completed_row = {
                "question": question,
                "document": document,
                "answer": result["answer"],
                "ground_truth": reference,
                "scores": scores,
                "sources": sources,
            }
            if row.get("qa_id"):
                completed_row["qa_id"] = row["qa_id"]

            outputs.append(completed_row)
            completed_ids.update(identifiers)
            save_checkpoint(checkpoint_path, outputs)

        except Exception as error:
            save_checkpoint(checkpoint_path, outputs)
            if is_rate_limit_error(error):
                print(
                    "\nGroq rate limit reached (HTTP 429 / daily token quota). "
                    "Completed rows were saved; the current question remains "
                    "pending for the next run."
                )
                interrupted = True
                break
            raise

        # -----------------------------------------------------
        # Display individual scores
        # -----------------------------------------------------
        print(
            "Scores:"
        )

        print(
            f"  Faithfulness      : "
            f"{format_score(scores['faithfulness'])}"
        )

        print(
            f"  Answer Relevancy   : "
            f"{format_score(scores['answer_relevancy'])}"
        )

        print(
            f"  Context Precision  : "
            f"{format_score(scores['context_precision'])}"
        )

        print(
            f"  Context Recall     : "
            f"{format_score(scores['context_recall'])}"
        )

        # -----------------------------------------------------
        # Verify retrieved documents
        # -----------------------------------------------------
        retrieved_documents = sorted(
            {
                source["document"]
                for source in sources
            }
        )

        print(
            "Retrieved documents:"
        )

        for retrieved_document in retrieved_documents:
            print(
                f"  - {retrieved_document}"
            )

    remaining = len(rows) - len(outputs) - skipped
    return outputs, skipped, remaining, interrupted


# =============================================================
# Run evaluation
# =============================================================
def run_evaluation():

    # ---------------------------------------------------------
    # Evaluation dataset
    # ---------------------------------------------------------
    path = (
        SETTINGS.eval_dir
        / "questions.jsonl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Create {path} before running evaluation. "
            "Use questions.example.jsonl as the template."
        )

    # ---------------------------------------------------------
    # Check Groq API key
    # ---------------------------------------------------------
    if (
        not SETTINGS.groq_api_key
        or SETTINGS.groq_api_key.startswith(
            "PASTE_"
        )
    ):
        raise RuntimeError(
            "Set GROQ_API_KEY in .env "
            "before running evaluation."
        )

    # ---------------------------------------------------------
    # Load questions
    # ---------------------------------------------------------
    rows = load_jsonl(
        path
    )

    print(
        "=" * 60
    )

    print(
        "POLICY DOCUMENT QA & EXTRACTION ENGINE"
    )

    print(
        "RAGAS EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Evaluation questions: {len(rows)}"
    )

    # ---------------------------------------------------------
    # Load the checkpoint before loading the pipeline so a
    # completed evaluation does not initialize models again.
    # ---------------------------------------------------------
    out_dir = (
        SETTINGS.root
        / "evals"
    )

    out_dir.mkdir(
        exist_ok=True
    )

    out = (
        out_dir
        / "latest_results.json"
    )

    outputs = load_checkpoint(out)
    completed_ids = {
        key
        for row in outputs
        for key in row_match_keys(row)
    }
    pending_rows = [
        row
        for row in rows
        if not row_match_keys(row) & completed_ids
    ]
    skipped = len(rows) - len(pending_rows)

    print(f"Completed questions : {len(outputs)}")
    print(f"Skipped questions   : {skipped}")
    print(f"Remaining questions : {len(pending_rows)}")

    interrupted = False
    if pending_rows:
        print(
            "\nLoading RAG pipeline..."
        )

        pipeline = RAGPipeline.from_disk()

        print(
            "RAG pipeline loaded."
        )

        outputs, skipped_during_run, _, interrupted = asyncio.run(
            score_rows(
                pending_rows,
                pipeline,
                outputs,
                out,
            )
        )
        skipped += skipped_during_run

    checkpoint = save_checkpoint(out, outputs)
    summary = checkpoint["summary"]
    remaining = len(rows) - len(outputs)

    # =========================================================
    # Final summary
    # =========================================================
    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Questions evaluated : "
        f"{len(outputs)}"
    )

    print(
        f"Skipped questions   : "
        f"{skipped}"
    )

    print(
        f"Remaining questions : "
        f"{remaining}"
    )

    if summary["faithfulness"] is not None:
        print(
            f"Faithfulness        : "
            f"{summary['faithfulness']:.4f}"
        )
    else:
        print(
            "Faithfulness        : N/A"
        )

    if summary["answer_relevancy"] is not None:
        print(
            f"Answer Relevancy     : "
            f"{summary['answer_relevancy']:.4f}"
        )
    else:
        print(
            "Answer Relevancy     : N/A"
        )

    if summary["context_precision"] is not None:
        print(
            f"Context Precision    : "
            f"{summary['context_precision']:.4f}"
        )
    else:
        print(
            "Context Precision    : N/A"
        )

    if summary["context_recall"] is not None:
        print(
            f"Context Recall       : "
            f"{summary['context_recall']:.4f}"
        )
    else:
        print(
            "Context Recall       : N/A"
        )

    print(
        "=" * 60
    )

    print(
        "\nDetailed results saved to:"
    )

    print(
        out
    )

    if interrupted:
        print(
            "Resume later to continue with the first pending question."
        )


# =============================================================
# Main
# =============================================================
if __name__ == "__main__":
    run_evaluation()