import json
from glob import glob


def _sanity_check_exams_corrections_lengths(all_exams, all_corrections):
    for exam, correction in zip(all_exams, all_corrections):
        assert len(exam["questions"]) == len(correction["questions"])


def _sanity_check_exams_corrections_match(all_exams, all_corrections):
    for exam, correction in zip(all_exams, all_corrections):
        for question, correction in zip(exam["questions"], correction["questions"]):
            assert question["number"] == correction["question_number"], (
                question["number"],
                correction["question_number"],
                exam["preamble"]["year"],
                exam["preamble"]["phase"],
                exam["preamble"]["subject"],
            )
            assert question["group"] == correction["question_group"], (
                question["group"],
                correction["question_group"],
                exam["preamble"]["year"],
                exam["preamble"]["phase"],
                exam["preamble"]["subject"],
            )


def _sanity_check_mcq(all_exams, all_corrections):
    for exam, correction in zip(all_exams, all_corrections):
        for question, corr_question in zip(exam["questions"], correction["questions"]):
            if question["type"] in ["multiple-choice"]:
                try:
                    assert corr_question["correction_instructions"]["correct_answer"] is not None
                except Exception:
                    assert False, (corr_question, exam["preamble"])


def load_all_files():
    all_exams, all_corrections = [], []
    dataset_dir = "dataset_json"

    print(f"Loading dataset from {dataset_dir}")
    for filename in sorted(glob(f"{dataset_dir}/*.json")):
        if "all.json" in filename:
            continue

        with open(filename, "r") as fin:
            questions = json.load(fin)

        for q in questions:
            exam_data = {
                "preamble": {
                    "year": q["year"],
                    "phase": q["phase"],
                    "subject": q["subject"],
                },
                "questions": [
                    {
                        "number": q["question_number"],
                        "group": q["question_group"],
                        "question": q["question"],
                        "type": "multiple-choice",
                        "options": q["choices"],
                        "refers-to-documents": [],
                        "refers-to-images": [],
                    }
                ],
                "documents": [],
                "filename": filename,
            }
            correction_data = {
                "questions": [
                    {
                        "question_number": q["question_number"],
                        "question_group": q["question_group"],
                        "correction_instructions": {"correct_answer": chr(ord("A") + q["answer"])},
                    }
                ]
            }
            all_exams.append(exam_data)
            all_corrections.append(correction_data)

    return (all_exams, all_corrections)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    args = p.parse_args()

    all_exams, all_corrections = load_all_files()
    _sanity_check_exams_corrections_lengths(all_exams, all_corrections)
    _sanity_check_exams_corrections_match(all_exams, all_corrections)
    _sanity_check_mcq(all_exams, all_corrections)
