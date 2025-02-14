"""
Relevance judgment processor for UMBRELA-style evaluation.
Implements the Bing Relevance Assessment methodology.
"""

import json
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional
from relevance_scoring import grade_each_pq_pair

class RelevanceProcessor:
    """Base class for processing relevance judgments using UMBRELA methodology."""
    
    def __init__(self, result_path: str):
        self.result_path = Path(result_path)
        self.setup_paths()
            
    def setup_paths(self):
        """
        Setup file paths for results and logging.
        Creates necessary directories for storing outputs.
        """
        # Define paths relative to result path
        self.generation_path = self.result_path.parent / "generation_errors" / self.result_path.name
        self.logs_path = self.result_path.parent / "logs" / self.result_path.name.replace(".txt", ".jsonl")
        self.cuda_errors_path = self.result_path.parent / "cuda_errors" / self.result_path.name
        
        # Create parent directories for each path
        for path in [self.generation_path, self.logs_path, self.cuda_errors_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
                
    def debug_print(self, qidx, docidx, final_score, scoring_log):
        """Debugging prints for first run."""
        if not hasattr(self, "first_run_complete"):
            self.first_run_complete = True
            print("\n=== First Run Debug Info ===")
            print(f"Query ID: {qidx}")
            print(f"Doc ID: {docidx}")
            print(f"Model Output: {scoring_log['LLMs_output']}")
            print(f"Final Score: {final_score}")
            print("="*30)

def grade_pq_pairs(test_qrel, docid_to_doc, qid_to_query, result_path: str, 
                  pipeline, system_message: str, mode: str = "zeroshot_bing", max_pairs: Optional[int] = None):
    """
    Process relevance judgments using UMBRELA methodology.
    
    Args:
        test_qrel: DataFrame containing query-document pairs to evaluate
        docid_to_doc: Dictionary mapping document IDs to document text
        qid_to_query: Dictionary mapping query IDs to query text
        result_path: Path to save results
        pipeline: Model pipeline for inference
        system_message: System message (empty for UMBRELA)
        mode: UMBRELA prompt mode (default: "zeroshot_bing")
    """
    processor = RelevanceProcessor(result_path)
    total_pairs = len(test_qrel) if max_pairs is None else min(len(test_qrel), max_pairs)
    print(f"Processing {total_pairs} pairs out of {len(test_qrel)} total pairs")

    with open(processor.result_path, 'w') as result_file, \
         open(processor.generation_path, 'w') as generation_errors_file, \
         open(processor.cuda_errors_path, 'w') as cuda_errors_file:
        
        for idx, eachline in enumerate(tqdm(test_qrel.itertuples(index=True), total=total_pairs)):
            if max_pairs is not None and idx >= max_pairs:
                print(f"\nReached maximum pairs limit ({max_pairs})")
                break
            
            qidx = eachline.qid
            docidx = eachline.docid

            try:
                # Handle document ID type conversion
                try:
                    final_score, scoring_log = grade_each_pq_pair(
                        query=qid_to_query[qidx],
                        passage=docid_to_doc[docidx],
                        pipeline=pipeline,
                        log_file_path=processor.logs_path,
                        system_message=system_message,
                        qidx=qidx,
                        docidx=docidx,
                        mode=mode
                    )
                except KeyError:
                    # Try with string conversion
                    docidx = str(docidx)
                    final_score, scoring_log = grade_each_pq_pair(
                        query=qid_to_query[qidx],
                        passage=docid_to_doc[docidx],
                        pipeline=pipeline,
                        log_file_path=processor.logs_path,
                        system_message=system_message,
                        qidx=qidx,
                        docidx=docidx,
                        mode=mode
                    )
                # Debug print for first run
                processor.debug_print(qidx, docidx, final_score, scoring_log)

                # Write results in TREC format
                if isinstance(final_score, int) and 0 <= final_score <= 3:
                    result_file.write(f"{qidx} 0 {docidx} {final_score}\n")
                else:
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                    generation_errors_file.write(f"Invalid score: {qidx} 0 {docidx} {final_score}\n")

            except Exception as e:
                # Log any errors and continue processing
                cuda_errors_file.write(f"{qidx} {docidx}: {str(e)}\n")
                result_file.write(f"{qidx} 0 {docidx} 0\n")
                print(f"Error processing {qidx}, {docidx}: {str(e)}")