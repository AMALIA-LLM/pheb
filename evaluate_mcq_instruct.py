import argparse
import json
import os
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging
from vllm import LLM, SamplingParams

from loader import load_all_files

logging.set_verbosity_error()


def build_prompt(
    exam_q,
    format_type="boxed",
    eval_method="generate",
    prepend_answer_prefix=False,
    use_generation_prompt=False,
):
    """Build an instruction prompt for a multiple-choice question.

    format_type:
        "boxed"       – chain-of-thought, answer in \\boxed{X}
        "boxed_only"  – answer only in \\boxed{X}
        "letter_only" – answer with a single letter A/B/C/D
        "number_only" – answer with a single number 1/2/3/4

    eval_method:
        "generate"    – model generates a response
        "likelihood"  – compare next-token logits for each option label

    prepend_answer_prefix:
        If True, the chat prompt ends with "A resposta é" / "The answer is"
        so the model's next token is the answer letter.
    """
    options = exam_q["options"]

    question_text = exam_q["question"]
    full_question = ("Pergunta:\n") + question_text + "\n"

    for i, option in enumerate(options):
        label = str(i + 1) if format_type == "number_only" else chr(ord("A") + i)
        full_question += f"{label}) {option}\n"

    if eval_method == "likelihood" and not use_generation_prompt:
        instruction = ""
    elif eval_method == "generate" and prepend_answer_prefix:
        instruction = ""
    elif format_type == "boxed":
        instruction = (
            "Escolha a opção correta para a pergunta acima. Apresente o seu raciocínio antes de responder. "
            "No final, apresente a sua resposta no formato \\boxed{X}, onde X é a letra da opção correta (A, B, C ou D)."
        )
    elif format_type == "boxed_only":
        instruction = (
            "Escolha a opção correta para a pergunta acima. Responda apenas no formato \\boxed{X}, "
            "onde X é a letra da opção correta (A, B, C ou D)."
        )
    elif format_type == "letter_only":
        instruction = (
            "Escolha a opção correta para a pergunta acima. Responda apenas com a letra da opção correta (A, B, C ou D). Não inclua nenhum outro texto."
        )
    elif format_type == "number_only":
        instruction = (
            "Escolha a opção correta para a pergunta acima. Responda apenas com o número da opção correta (1, 2, 3 ou 4). Não inclua nenhum outro texto."
        )
    else:
        raise ValueError(f"Unknown format_type: {format_type}")

    return full_question + instruction, options


def extract_answer(response, format_type="boxed"):
    """Extract the predicted answer letter from a model response. It's unfortunate, but we sometimes need all these cases."""
    if format_type in ("boxed", "boxed_only"):
        m = re.search(r"\\boxed{([A-D])}", response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"([A-D])\.?\s*$", response.strip())
        if m:
            return m.group(1).upper()
        matches = re.findall(r"\b([A-D])\b", response)
        if matches:
            return matches[-1].upper()

    elif format_type == "letter_only":
        m = re.search(r"^([A-D])\b|([A-D])$", response.strip())
        if m:
            return (m.group(1) or m.group(2)).upper()
        m = re.search(r"\b([A-D])\b", response)
        if m:
            return m.group(1).upper()

    elif format_type == "number_only":
        m = re.search(r"^([1-4])\b|([1-4])$", response.strip())
        if m:
            return chr(ord("A") + int(m.group(1) or m.group(2)) - 1)
        m = re.search(r"\b([1-4])\b", response)
        if m:
            return chr(ord("A") + int(m.group(1)) - 1)

    return None


def letter_to_index(letter):
    try:
        return ord(letter.upper()) - ord("A")
    except (TypeError, AttributeError):
        return -1


def evaluate(
    model_checkpoint,
    custom_system_prompt="",
    format_type="boxed",
    eval_method="generate",
    prepend_answer_prefix=False,
    use_generation_prompt=False,
    output_dir="mcq-results",
):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(model=model_checkpoint, dtype="bfloat16", max_model_len=9654,
              limit_mm_per_prompt={"image": 0})

    all_exams, all_corrections = load_all_files()

    questions_data = []
    for exam, correction in zip(all_exams, all_corrections):
        exam_info = {
            "year": exam["preamble"]["year"],
            "phase": exam["preamble"]["phase"],
            "subject": exam["preamble"]["subject"],
        }
        for exam_q, cc_q in zip(exam["questions"], correction["questions"]):
            if exam_q["type"] != "multiple-choice":
                continue

            prompt, options = build_prompt(
                exam_q, format_type, eval_method, prepend_answer_prefix, use_generation_prompt
            )

            correct_letter = cc_q["correction_instructions"]["correct_answer"]
            if letter_to_index(correct_letter) == -1:
                print(f"Skipping {exam['filename']}: invalid correct answer '{correct_letter}'")
                continue

            messages = []
            if custom_system_prompt:
                messages.append({"role": "system", "content": custom_system_prompt})
            messages.append({"role": "user", "content": prompt})

            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if prepend_answer_prefix:
                input_text += "A resposta é"

            questions_data.append({
                "exam_info": exam_info,
                "input_text": input_text,
                "prompt": prompt,
                "options": options,
                "correct_letter": correct_letter,
                "exam_q": exam_q,
            })

    if eval_method == "likelihood":
        sampling_params = SamplingParams(max_tokens=1, logprobs=20, temperature=0)
    else:
        max_new_tokens = 2048 if "boxed" in format_type else 10
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    print(f"Running inference on {len(questions_data)} questions...", flush=True)
    prompts = [q["input_text"] for q in questions_data]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for qdata, output in tqdm(zip(questions_data, outputs), total=len(questions_data)):
        exam_info = qdata["exam_info"]
        exam_q = qdata["exam_q"]
        options = qdata["options"]
        correct_letter = qdata["correct_letter"]

        if eval_method == "likelihood":
            if format_type == "number_only":
                option_chars = [str(i + 1) for i in range(len(options))]
                option_labels = option_chars
            else:
                option_chars = [chr(ord("A") + i) for i in range(len(options))]
                option_labels = option_chars

            token_logprobs = output.outputs[0].logprobs[0]  # dict {token_id: Logprob}

            def best_token_logprob(char):
                # Try both with and without leading space; use whichever is in the logprobs dict.
                ids_with = tokenizer.encode(f" {char}", add_special_tokens=False)
                ids_without = tokenizer.encode(char, add_special_tokens=False)
                tid_with = ids_with[-1]
                tid_without = ids_without[-1]
                if tid_with in token_logprobs:
                    return token_logprobs[tid_with].logprob
                if tid_without in token_logprobs:
                    return token_logprobs[tid_without].logprob
                return float("-inf")

            option_logits = [best_token_logprob(c) for c in option_chars]

            predicted_idx = option_logits.index(max(option_logits))
            predicted_letter = chr(ord("A") + predicted_idx)
            response = f"[Likelihood] {dict(zip(option_labels, option_logits))}"
            is_correct = predicted_letter == correct_letter

            print(f"Logprobs: {dict(zip(option_labels, option_logits))}")
            print(f"Predicted: {predicted_letter}, Correct: {correct_letter}, Is Correct: {is_correct}", flush=True)

        else:  # generate
            response = output.outputs[0].text
            predicted_letter = extract_answer(response, format_type)
            is_correct = (predicted_letter == correct_letter) if predicted_letter else False

            print(response, "\n", predicted_letter, correct_letter, is_correct, flush=True)

        results.append({
            **exam_info,
            "question_number": exam_q["number"],
            "question_group": exam_q["group"],
            "question_text": exam_q["question"],
            "prompt": qdata["prompt"],
            "model_response": response,
            "predicted_answer": predicted_letter,
            "correct_answer": correct_letter,
            "is_correct": is_correct,
            "format_type": format_type,
            "eval_method": eval_method,
            "prepend_answer_prefix": prepend_answer_prefix,
        })

    os.makedirs(output_dir, exist_ok=True)
    model_name = model_checkpoint.replace("/", "__")
    eval_suffix = f"_{format_type}_{eval_method}" + ("_prefix" if prepend_answer_prefix else "")
    output_file = f"{output_dir}/instruction_mcq_results_{model_name}{eval_suffix}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    answered = [r for r in results if r["predicted_answer"] is not None]
    correct = [r for r in results if r["is_correct"]]
    print(f"\nResults saved to {output_file}")
    print(f"Total: {len(results)} | Answered: {len(answered)} ({len(answered)/len(results):.1%}) | Correct: {len(correct)} ({len(correct)/len(results):.1%})")

    subjects = sorted({r["subject"] for r in results})
    print("\nPer-subject results:")
    for subject in subjects:
        sub = [r for r in results if r["subject"] == subject]
        sub_correct = sum(r["is_correct"] for r in sub)
        print(f"  {subject}: {sub_correct}/{len(sub)} ({sub_correct/len(sub):.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate instruction-tuned models on PHEB multiple-choice questions.")
    parser.add_argument("--model_list", nargs="+", required=True, help="HuggingFace model checkpoints to evaluate.")
    parser.add_argument("--custom_system_prompt", type=str, default="", help="Optional system prompt.")
    parser.add_argument(
        "--format_type", default="boxed", choices=["boxed", "boxed_only", "letter_only", "number_only"],
        help="Response format. 'boxed': CoT with \\boxed{X}; 'letter_only': just A/B/C/D.",
    )
    parser.add_argument(
        "--eval_method", default="generate", choices=["generate", "likelihood"],
        help="'generate': model generates a response; 'likelihood': compare next-token logits.",
    )
    parser.add_argument(
        "--prepend_answer_prefix", action="store_true",
        help="Append 'A resposta é'/'The answer is' before the answer token (recommended for likelihood eval).",
    )
    parser.add_argument(
        "--use_generation_prompt", action="store_true",
        help="Use full generation instructions even in likelihood mode.",
    )
    parser.add_argument("--output_dir", default="mcq-results", help="Directory to save result JSON files.")
    args = parser.parse_args()

    for model_checkpoint in args.model_list:
        print(f"\nEvaluating {model_checkpoint} | format={args.format_type} | method={args.eval_method}", flush=True)
        evaluate(
            model_checkpoint,
            custom_system_prompt=args.custom_system_prompt,
            format_type=args.format_type,
            eval_method=args.eval_method,
            prepend_answer_prefix=args.prepend_answer_prefix,
            use_generation_prompt=args.use_generation_prompt,
            output_dir=args.output_dir,
        )
