import numpy as np
import collections
import evaluate
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def predict_answers_and_evaluate(start_logits, end_logits, logits, eval_set, examples):
    """
    make predictions
    Args:
    start_logits : start_position prediction logits
    end_logits: end_position prediction logits
    eval_set: processed val data
    examples: unprocessed val data with context text
    """
    # appending all id's corresponding to the base context id
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(eval_set):
        example_to_features[feature["base_id"]].append(idx)

    class_dict = {'UNCLEAR': 0, 'HIGH': 0, 'LOW': 1}
    n_best = 20
    max_answer_length = 167
    predicted_answers = []

    true_cls_labels = []
    pred_cls_labels = []
    pred_cls_labels_dict = {}
    prob_cls_dict = {}

    for example in tqdm(examples, total=len(examples), desc="Predicting answers"):
        example_id = example["id"]
        context = example["context"]
        true_cls_labels.append(class_dict[example["label"]])
        cls_prediction = (logits[example_to_features[example_id][0]] >= 0.5)

        pred_cls_labels_dict[example_id] = 1 if cls_prediction[0] else 0
        pred_cls_labels.append(1 if cls_prediction[0] else 0)
        prob_cls_dict[example_id] = logits[example_to_features[example_id][0]][0].astype(float)

        answers = []

        # looping through each sub contexts corresponding to a context and finding
        # answers
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]

            # sorting the predictions of all hidden states and taking best n_best prediction
            # means taking the index of top 20 tokens
            start_indexes = np.argsort(start_logit).tolist()[::-1][:n_best]
            end_indexes = np.argsort(end_logit).tolist()[::-1][:n_best]

            for start_index in start_indexes:
                for end_index in end_indexes:

                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    text = context[offsets[start_index][0]: offsets[end_index][1]]
                    # workaround for skipping extra terms in the answer
                    _new_answer_ = text.split('.')
                    if len(_new_answer_) > 1 and len(_new_answer_[1].split()) == 1:
                        text = _new_answer_[0] + '.'

                    answers.append({
                        "text": text,
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })

            # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer['text']}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    metric = evaluate.load("squad")
    cls_f1 = f1_score(true_cls_labels, pred_cls_labels, average='macro', zero_division=0)
    cls_pr = precision_score(true_cls_labels, pred_cls_labels, average='macro', zero_division=0)
    cls_rc = recall_score(true_cls_labels, pred_cls_labels, average='macro', zero_division=0)
    cls_ac = accuracy_score(true_cls_labels, pred_cls_labels)

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]

    metric_ = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    return predicted_answers, metric_, cls_f1, cls_pr, cls_rc, cls_ac, pred_cls_labels_dict, prob_cls_dict
