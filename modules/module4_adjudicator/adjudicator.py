# adjudicator.py

import ollama
from prompts import build_pass1_prompt, build_pass2_prompt, build_pass3_prompt

MODEL_NAME = "llama3"

def run_llm(prompt):
    """Send a prompt to the local Ollama server and return the response text."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


def constitutional_adjudicate(article_text, module1_output, evidence_score, module3_output):
    print("\n=== PASS 1: Generating Initial Verdict ===")
    pass1_prompt = build_pass1_prompt(article_text, module1_output, evidence_score, module3_output)
    initial_verdict = run_llm(pass1_prompt)
    print(initial_verdict)

    print("\n=== PASS 2: Constitutional Self-Critique ===")
    pass2_prompt = build_pass2_prompt(initial_verdict)
    critique = run_llm(pass2_prompt)
    print(critique)

    print("\n=== PASS 3: Revised Final Verdict ===")
    pass3_prompt = build_pass3_prompt(initial_verdict, critique)
    final_verdict = run_llm(pass3_prompt)
    print(final_verdict)

    return {
        "initial_verdict": initial_verdict,
        "critique":        critique,
        "final_verdict":   final_verdict
    }