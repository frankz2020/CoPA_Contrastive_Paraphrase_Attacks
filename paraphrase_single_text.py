#!/usr/bin/env python3
"""
CoPA Single Text Paraphraser
A standalone script to paraphrase a single AI-generated text using the CoPA method.

Usage:
    python paraphrase_single_text.py --text "Your AI-generated text here"
    
    Or from a file:
    python paraphrase_single_text.py --input_file input.txt --output_file output.txt
"""

import torch
import argparse
from scripts.model import load_tokenizer, load_model
from scripts.sim_model import load_sim_model, FileSim, batcher
from sacremoses import MosesTokenizer
from argparse import Namespace

# System prompts (from the paper)
SYSTEM_PROMPT_HUMAN = "You are a helpful paraphraser. You are given an input passage 'INPUT'. You should paraphrase 'INPUT' to print 'OUTPUT'. 'OUTPUT' should preserve the meaning and content of 'INPUT'. 'OUTPUT' should not be very shorter than 'INPUT'."
SYSTEM_PROMPT_MACHINE = "You are a helpful assistant."

ASSISTANT_SEPERATOR = '\nassistant\n'

class CopaParaphraser:
    def __init__(self, 
                 base_model_name="qwen2.5-72b",
                 lamda=0.5,
                 alpha=1e-5,
                 temperature=1.0,
                 max_tries=10,
                 device="cuda",
                 cache_dir="../cache"):
        """
        Initialize the CoPA paraphraser.
        
        Args:
            base_model_name: The LLM to use (default: qwen2.5-72b)
            lamda: Contrast strength parameter (default: 0.5)
            alpha: Cutoff threshold (default: 1e-5)
            temperature: Sampling temperature (default: 1.0)
            max_tries: Max attempts to get good paraphrase (default: 10)
            device: Device to use (default: cuda)
            cache_dir: Cache directory for models (default: ../cache)
        """
        print(f"Loading tokenizer for {base_model_name}...")
        self.tokenizer = load_tokenizer(base_model_name, 'xsum', cache_dir)
        
        print(f"Loading model {base_model_name}... (this may take a while)")
        self.model = load_model(base_model_name, device, cache_dir)
        
        print("Loading SIM model for semantic similarity checking...")
        self.sim_model = load_sim_model('paraphrase-at-scale/model.para.lc.100.pt')
        self.sim_model.eval()
        
        self.lamda = lamda
        self.alpha = alpha
        self.temperature = temperature
        self.max_tries = max_tries
        self.device = device
        
        # Prompts (you can customize these)
        self.prompt_human = "Rewrite the following INPUT in the tone of a text message to a friend without any greetings or emojis:"
        self.prompt_machine = "Repeat the following paragraph:"
        
        # Setup for similarity scoring
        self.sim_args = Namespace(
            batch_size=32, 
            entok=MosesTokenizer(lang='en'), 
            sp=self.sim_model.sp,
            model=self.sim_model, 
            lower_case=self.sim_model.args.lower_case,
            tokenize=self.sim_model.args.tokenize
        )
        self.sim_scorer = FileSim()
        
        print("✓ CoPA paraphraser initialized successfully!")
    
    def paraphrase(self, text, similarity_threshold=0.76, verbose=True):
        """
        Paraphrase a single text using CoPA.
        
        Args:
            text: The AI-generated text to paraphrase
            similarity_threshold: Minimum similarity score (default: 0.76)
            verbose: Print progress information
            
        Returns:
            The paraphrased text
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Starting CoPA paraphrasing...")
            print(f"{'='*60}")
            print(f"Input text length: {len(text.split())} words")
            print(f"Parameters: λ={self.lamda}, α={self.alpha}, temp={self.temperature}")
            print(f"{'='*60}\n")
        
        # Prepare human-like prompt
        messages_human = [
            {"role": "system", "content": SYSTEM_PROMPT_HUMAN},
            {"role": "user", "content": f'{self.prompt_human} {text}'}
        ]
        
        # Prepare machine-like prompt
        messages_machine = [
            {"role": "system", "content": SYSTEM_PROMPT_MACHINE},
            {"role": "user", "content": f'{self.prompt_machine} {text}'}
        ]
        
        # Apply chat template
        text_human = self.tokenizer.apply_chat_template(
            messages_human,
            tokenize=False,
            add_generation_prompt=True
        )
        text_machine = self.tokenizer.apply_chat_template(
            messages_machine,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs_human = self.tokenizer([text_human], return_tensors="pt").to(self.device)
        model_inputs_machine = self.tokenizer([text_machine], return_tensors="pt").to(self.device)
        
        # Generation parameters
        model_kwargs = {
            "max_new_tokens": 512,
            "temperature": self.temperature,
            "model_inputs_machine": model_inputs_machine,
            "lamda": self.lamda,
            "alpha": self.alpha
        }
        
        # Try multiple times to get a good paraphrase
        best_score = 0
        best_paraphrase = ""
        
        for attempt in range(1, self.max_tries + 1):
            if verbose:
                print(f"Attempt {attempt}/{self.max_tries}...", end=" ", flush=True)
            
            # Generate with contrastive decoding
            generated_ids = self.model.generate(
                **model_inputs_human,
                **model_kwargs
            )
            
            # Decode
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract the assistant's response
            if ASSISTANT_SEPERATOR in response:
                paraphrased = response.split(ASSISTANT_SEPERATOR)[1]
            else:
                # Fallback if separator not found
                paraphrased = response
            
            # Calculate semantic similarity
            score = self.sim_scorer.score(self.sim_args, batcher, paraphrased, text)[0]
            
            if verbose:
                print(f"Similarity: {score:.4f}, Length: {len(paraphrased.split())} words")
            
            # Keep track of best result
            if score > best_score:
                best_score = score
                best_paraphrase = paraphrased
            
            # If we meet threshold, we're done
            if score > similarity_threshold:
                if verbose:
                    print(f"\n✓ Success! Achieved similarity: {score:.4f}")
                break
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Best similarity achieved: {best_score:.4f}")
            print(f"Output text length: {len(best_paraphrase.split())} words")
            print(f"{'='*60}\n")
        
        return best_paraphrase


def main():
    parser = argparse.ArgumentParser(
        description="CoPA: Paraphrase AI-generated text to evade detection"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Text to paraphrase (as string)")
    input_group.add_argument("--input_file", type=str, help="File containing text to paraphrase")
    
    # Output options
    parser.add_argument("--output_file", type=str, help="Save paraphrased text to file")
    
    # CoPA parameters
    parser.add_argument("--base_model", type=str, default="qwen2.5-72b",
                       help="Base model to use (default: qwen2.5-72b)")
    parser.add_argument("--lamda", type=float, default=0.5,
                       help="Contrast strength (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=1e-5,
                       help="Cutoff threshold (default: 1e-5)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (default: 1.0)")
    parser.add_argument("--max_tries", type=int, default=10,
                       help="Max paraphrasing attempts (default: 10)")
    parser.add_argument("--similarity_threshold", type=float, default=0.76,
                       help="Minimum similarity threshold (default: 0.76)")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--cache_dir", type=str, default="../cache",
                       help="Cache directory (default: ../cache)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress messages")
    
    args = parser.parse_args()
    
    # Load input text
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_text = f.read().strip()
        print(f"Loaded text from: {args.input_file}")
    else:
        input_text = args.text
    
    if not input_text:
        print("Error: Input text is empty!")
        return
    
    # Initialize paraphraser
    paraphraser = CopaParaphraser(
        base_model_name=args.base_model,
        lamda=args.lamda,
        alpha=args.alpha,
        temperature=args.temperature,
        max_tries=args.max_tries,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    # Paraphrase
    paraphrased_text = paraphraser.paraphrase(
        input_text,
        similarity_threshold=args.similarity_threshold,
        verbose=not args.quiet
    )
    
    # Output
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(paraphrased_text)
        print(f"\n✓ Paraphrased text saved to: {args.output_file}")
    else:
        print("\n" + "="*60)
        print("PARAPHRASED TEXT:")
        print("="*60)
        print(paraphrased_text)
        print("="*60)


if __name__ == "__main__":
    main()

