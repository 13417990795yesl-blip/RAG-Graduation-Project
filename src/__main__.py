"""List and launch existing RAG experiments without eagerly loading models."""

import argparse
import runpy
import sys


COMMANDS = {
    "build-chronoqa": ("build_chronoqa_mini", "Build the ChronoQA mini dataset"),
    "build-hotpotqa": ("build_hotpotqa_mini", "Build the HotpotQA mini dataset"),
    "build-multihoprag": ("build_multihoprag_mini", "Build the MultiHop-RAG mini dataset"),
    "index-multihoprag": ("build_multihoprag_no_metadata_vector_db", "Build the no-metadata Chroma collection"),
    "retrieve-chronoqa": ("run_chronoqa_retrieval_eval", "ChronoQA dense and pointwise retrieval"),
    "retrieve-hotpotqa": ("run_hotpotqa_retrieval_eval", "HotpotQA dense and pointwise retrieval"),
    "retrieve-multihoprag": ("run_multihoprag_retrieval_eval", "Original MultiHop-RAG retrieval experiment"),
    "rerank-multihoprag": ("run_multihoprag_llm_resume_eval", "Resumable LLM reranking; uses API"),
    "generate-multihoprag": ("generate_multihoprag_english", "Generate and evaluate English answers; uses API"),
    "compare-multihoprag": ("run_multihoprag_baseline_pointwise_generation", "Baseline / pointwise / ColBERT plus generation; uses GPU and API"),
    "extra-multihoprag": ("run_multihoprag_extra_retrieval_eval", "Original extended retrieval experiment; uses API"),
    "diagnose-multihoprag": ("diagnose_multihoprag_llm_top20", "Small LLM-output diagnostic; uses API"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    parser.add_argument("--list", action="store_true", help="List commands without loading dependencies or models")
    args, remaining = parser.parse_known_args()
    if args.list:
        for command, (_, description) in COMMANDS.items():
            print(f"{command:24} {description}")
        return
    if args.command is None:
        parser.print_help()
        return
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    module = "src." + COMMANDS[args.command][0]
    sys.argv = [module, *remaining]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
