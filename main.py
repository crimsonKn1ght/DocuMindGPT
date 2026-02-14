"""
DocuMindGPT – Main Orchestrator
CLI entry-point with two modes:
  1. Upload Mode  – ingest a document into the vector store.
  2. Chat Mode    – ask questions against the ingested knowledge base.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from src.ingest import ingest_file
from src.rag_agent import query as rag_query
from src.eval_agent import evaluate

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DocuMindGPT")


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def _handle_upload(file_path: str) -> None:
    """Ingest a single file into Supabase."""
    logger.info("=== Upload Mode ===")
    try:
        count: int = ingest_file(file_path)
        logger.info("Done – %d chunks ingested from '%s'.", count, file_path)
    except Exception:
        logger.exception("Ingestion failed.")
        sys.exit(1)


def _handle_chat() -> None:
    """Interactive chat loop: query -> retrieve -> generate -> evaluate."""
    logger.info("=== Chat Mode === (type 'exit' or 'quit' to stop)")
    while True:
        try:
            user_input: str = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() in {"exit", "quit"}:
            logger.info("Exiting chat.")
            break

        try:
            # Retrieve + Generate
            answer, context = rag_query(user_input)

            print(f"\nAssistant: {answer}")

            # Evaluate (skip if no context was retrieved)
            if context:
                evaluation = evaluate(
                    user_query=user_input,
                    context=context,
                    answer=answer,
                )
                print(
                    f"\n[Eval] Score: {evaluation['score']}/10  |  "
                    f"Verdict: {evaluation['verdict']}\n"
                    f"       Reasoning: {evaluation['reasoning']}"
                )
            else:
                print("\n[Eval] Skipped – no context was retrieved.")

        except Exception:
            logger.exception("Error during chat turn.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="DocuMindGPT – Document-grounded Q&A with hallucination evaluation."
    )
    subparsers = parser.add_subparsers(dest="command")

    # upload sub-command
    upload_parser = subparsers.add_parser("upload", help="Ingest a document into the vector store.")
    upload_parser.add_argument("file", type=str, help="Path to a PDF or text file.")

    # chat sub-command
    subparsers.add_parser("chat", help="Interactive Q&A against the knowledge base.")

    args = parser.parse_args()

    if args.command == "upload":
        _handle_upload(args.file)
    elif args.command == "chat":
        _handle_chat()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
