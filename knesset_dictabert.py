
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def main():
    if len(sys.argv) < 3:
        print("usage: python <script.py> <path/to/masked_sentences.txt> <path/to/output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # create output dir
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model_name = "dicta-il/dictabert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    # read lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # output file
    output_file = os.path.join(output_dir, "dictabert_predictions.txt")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # original
            masked_sentence = line
            # replace
            masked_for_model = masked_sentence.replace("[*]", "[MASK]")

            # tokenize
            inputs = tokenizer.encode_plus(masked_for_model, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # find mask positions
            mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)

            if len(mask_token_index[1]) == 0:
                out_f.write(f"masked_sentence: {masked_sentence}\n")
                out_f.write(f"dictabert_sentence: {masked_sentence}\n")
                out_f.write(f"dictabert tokens: \n\n")
                continue

            # run model
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted_tokens = []
            for mask_pos in mask_token_index[1]:
                mask_logits = logits[0, mask_pos, :]
                top_token_id = torch.argmax(mask_logits)
                predicted_tokens.append(top_token_id.item())

            # convert to token strings
            predicted_tokens_str = tokenizer.convert_ids_to_tokens(predicted_tokens)

            # reconstruct
            reconstructed_sentence = masked_for_model
            for tok in predicted_tokens_str:
                reconstructed_sentence = reconstructed_sentence.replace("[MASK]", tok, 1)

            # write results
            out_f.write(f"masked_sentence: {masked_sentence}\n")
            out_f.write(f"dictabert_sentence: {reconstructed_sentence}\n")
            out_f.write(f"dictabert tokens: {','.join(predicted_tokens_str)}\n\n")

    print("done")

if __name__ == "__main__":
    main()
