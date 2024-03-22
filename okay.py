from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from jax import grad, jit, vmap


@jax.jit
def loss_fn(params, inputs, targets):
    logits = model(params, inputs)
    return jnp.mean((logits - targets)**2)

##jax model

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "es": "spanish",
    "fr": "french",
    "de": "german"
}



model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

with torch.no_grad():
    input_ids = torch.tensor(tokenizer.encode("The dog ate the ")).unsqueeze(0)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    # Get the most likely next token
    next_token_logits = logits[0, -1, :]
    next_token_id = next_token_logits.argmax().item()

    # Decode the token IDs to text
    generated_text = tokenizer.decode(input_ids[0].tolist() + [next_token_id])
    print(generated_text)



class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.Transformer(d_model=config.n_embd, nhead=config.n_head, num_encoder_layers=config.n_layer, num_decoder_layers=config.n_layer)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config



   def forward(self, input_ids, labels=None):
      transformer_outputs = self.transformer(input_ids)
      hidden_states = transformer_outputs[0]
      lm_logits = self.lm_head(hidden_states)
      outputs = (lm_logits,) + transformer_outputs[1:]  # add hidden states and attention if they are here
      if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (loss,) + outputs
      return outputs
