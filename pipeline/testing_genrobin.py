import evaluate
import torch
from tqdm import tqdm
from textstat import flesch_kincaid_grade
THRESHOLD = 0.5

def evaluate_model(ids, dataloader, model, tokenizer, args):

    model.eval()
    generated = []
    reference = []
    res = []

    with torch.no_grad():
        i = 0
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
            }
            answers = batch[2]
            labels = batch[4]
            outputs = model.predict(**inputs, max_length=args.max_seq_length)
            for output, target, input_text, label, prediction in zip(outputs["sequences"], answers, inputs['input_ids'], labels, outputs['prediction']):
                decoded_output = tokenizer.decode(output, skip_special_tokens=True)
                decoded_target = tokenizer.decode(target, skip_special_tokens=True)
                res.append({
                    'ids': ids[i],
                    'input': tokenizer.decode(input_text, skip_special_tokens=True),
                    'target': decoded_target,
                    'output': decoded_output,
                    'gold_label': label.detach().cpu().numpy().astype(int),
                    'predicted_label': (prediction > THRESHOLD).detach().cpu().numpy().astype(int)[0],
                    'probability': prediction.detach().cpu().numpy().astype(float)[0],
                })
                i += 1

                reference.append(decoded_target)
                generated.append(decoded_output)

    return res

