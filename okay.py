from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

with torch.no_grad():
    input_ids = torch.tensor(tokenizer.encode("The dog ate the apple")).unsqueeze(0)
    output_ids = input_ids

    for _ in range(5):  # Generate 5 tokens
        outputs = model(input_ids=output_ids)
        logits = outputs.logits

        # Get the most likely next token
        next_token_logits = logits[0, -1, :]
        next_token_id = next_token_logits.argmax().item()

        # Append the predicted token to the output sequence
        output_ids = torch.cat([output_ids, torch.tensor([[next_token_id]])], dim=1)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)