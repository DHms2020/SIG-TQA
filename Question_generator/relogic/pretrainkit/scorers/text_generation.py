import json
import torch
from nltk.translate.bleu_score import sentence_bleu


# from bleu import list_bleu


def is_rank_0():
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
    else:
        return True
    return False


class TextGenerationScorer:
    def __init__(self, tokenizer, bos_id, eos_id, output_path):
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.output_path = output_path
        self.tokenizer = tokenizer

    def __call__(self, prediction,beam_scores):
        preds = prediction.predictions
        preds_size = prediction.predictions_size
        label_ids = prediction.label_ids
        label_size = prediction.label_size
        p_start, l_start = 0, 0
        correct, total = 0, 0
        score = 0
        ref = []
        hyp = []
        if is_rank_0():
            fout = open(self.output_path, "w")
        for idx, (p_size, l_size) in enumerate(zip(preds_size, label_size)):
            p_end = p_start + p_size
            l_end = l_start + l_size
            pred = self.get_sequence(preds[p_start: p_end])
            label = self.get_sequence(label_ids[l_start: l_end])
            p_start = p_end
            l_start = l_end
            if pred == label:
                correct += 1
            total += 1

            # print("------展示beam scores------")
            # print(beam_scores[idx].item())

            if is_rank_0():
                pred_text = self.tokenizer.decode(pred, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True).strip()
                label_text = self.tokenizer.decode(label, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=True).strip()
                ref.append(label_text)
                hyp.append(pred_text)
                fout.write(
                    json.dumps({
                        "idx": idx,
                        "pred": pred_text,
                        "label": label_text,
                        "beam_score":beam_scores[idx].item()}) + "\n")
                score += sentence_bleu(label_text.strip().split(),pred_text.strip().split())
        # score = list_bleu([ref], hyp)
        # score = None
        return {
            "bleu": score / total,
            "accuracy": correct / total,
            "correct": correct,
            "total": total
        }

    def get_sequence(self, seq):
        processed_seq = []
        for idx in seq:
            if idx == self.bos_id:
                continue
            if idx == self.eos_id:
                break
            processed_seq.append(int(idx))
        return processed_seq
