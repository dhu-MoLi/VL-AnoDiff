import torch
from torch import nn
import torch.nn.functional as F
import os
import re
from functools import partial

# Based on EmbeddingManager, extended with LLM functionality
def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                              return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"
    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"
    token = token[0, 1]
    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]

class LLMEnhancedEmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            use_llm_enhancement=False,
            prompt_dir="prompts",
            sample_name=None,
            defect_name=None,
            llm_weight=0.5,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        self.string_to_param_dict = nn.ParameterDict()
        self.initial_embeddings = nn.ParameterDict()  # Base embeddings (not optimized)
        self.llm_embeddings = nn.ParameterDict()      # LLM embeddings (not optimized, used for regularization)

        # LLM enhancement parameters
        self.use_llm_enhancement = use_llm_enhancement
        self.prompt_dir = prompt_dir
        self.sample_name = sample_name
        self.defect_name = defect_name
        self.llm_weight = llm_weight
        
        self.progressive_words = progressive_words
        self.progressive_counter = 0
        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'):  # Stable Diffusion CLIP encoder
            self.is_clip = True
            self.tokenizer = embedder.tokenizer
            self.get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            self.get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            self.token_dim = 768
        else:  # LDM BERT encoder
            self.is_clip = False
            self.tokenizer = embedder.tknz_fn
            self.get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            self.get_embedding_for_tkn = embedder.transformer.token_emb
            self.token_dim = 1280

        # Initialize embedding parameters
        for idx, placeholder_string in enumerate(placeholder_strings):
            token = self.get_token_for_string(placeholder_string)

            # Initialize embedding parameters
            if initializer_words and idx < len(initializer_words):
                init_word = initializer_words[idx]
                init_word_token = self.get_token_for_string(init_word)

                with torch.no_grad():
                    init_word_embedding = self.get_embedding_for_tkn(init_word_token.cpu())

                # Create optimizable embedding parameters
                token_params = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), 
                    requires_grad=True
                )
                
                # Save initial embeddings for regularization
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), 
                    requires_grad=False
                )
                
                # If LLM enhancement is enabled, try to fetch LLM embeddings (sample_name only is sufficient for VISA)
                if self.use_llm_enhancement and self.sample_name:
                    llm_embeddings = self._get_llm_embeddings(
                        placeholder_string, num_vectors_per_token, init_word
                    )
                    
                    if llm_embeddings is not None:
                        # Blend initial embeddings with LLM embeddings
                        token_params = torch.nn.Parameter(
                            (1.0 - self.llm_weight) * token_params.data + 
                            self.llm_weight * llm_embeddings,
                            requires_grad=True
                        )
                        
                        # Save LLM embeddings for subsequent regularization
                        self.llm_embeddings[placeholder_string] = torch.nn.Parameter(
                            llm_embeddings, 
                            requires_grad=False
                        )
            else:
                # Random initialization when no initializer word is provided
                token_params = torch.nn.Parameter(
                    torch.rand(size=(num_vectors_per_token, self.token_dim)), 
                    requires_grad=True
                )
            
            # Register token and parameters with the manager
            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params
    
    def _get_llm_embeddings(self, placeholder_string, num_vectors, init_word):
        """Fetch embeddings from saved LLM prompts; returns None if no prompts are available."""
        prompts = self._read_prompts_from_file()
        if not prompts:
            print(f"No LLM prompts found for {self.sample_name}-{self.defect_name}")
            return None
            
        try:
            # Parse prompt text
            lines = prompts.strip().split('\n')
            # Check for an optimized prompt
            optimized_prompt = None
            for line in lines:
                if line.startswith("Optimized prompt:"):
                    optimized_prompt = line.replace("Optimized prompt:", "").strip()
                    break
            
            # Prefer optimized prompt when available
            if optimized_prompt:
                print(f"Using optimized prompt: {optimized_prompt}")
                prompts_to_use = [optimized_prompt] * 3  # Reuse the same optimized prompt
            else:
                # Otherwise use the original three prompts
                original_prompt = lines[1].strip()[1:-1]  # Strip brackets
                gen_prompt1 = lines[3].strip()[1:-1]
                gen_prompt2 = lines[4].strip()[1:-1]
                prompts_to_use = [original_prompt, gen_prompt1, gen_prompt2]
            # # Print found prompts for debugging
            # print(f"Found LLM prompts for {self.sample_name}-{self.defect_name}:")
            # print(f"  Original: {original_prompt}")
            # print(f"  Gen 1: {gen_prompt1}")
            # print(f"  Gen 2: {gen_prompt2}")
            print(f"Using prompts: {prompts_to_use}")
            
            # Convert prompts to embeddings
            embeddings = []
            
            # Process original and generated prompts
            for prompt in prompts_to_use:
                # CLIP encoder path
                if self.is_clip:
                    # Encode the full prompt
                    tokens = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=77, 
                        padding="max_length"
                    )["input_ids"]
                    
                    # Average embeddings over all non-padding tokens
                    with torch.no_grad():
                        token_embeddings = []
                        for i in range(1, tokens.shape[1]-1):  # Skip BOS and EOS tokens
                            if tokens[0, i] == 49407:  # Skip padding token
                                break
                            token_embeddings.append(self.get_embedding_for_tkn(tokens[0, i].cpu()))
                        
                        if token_embeddings:
                            # Compute mean embedding
                            embedding = torch.stack(token_embeddings).mean(dim=0)
                            embeddings.append(embedding)
                else:
                    # BERT encoder implementation (adjust as needed)
                    pass
            
            # Ensure we have enough embedding vectors
            if embeddings:
                # Repeat the last embedding if we have too few
                while len(embeddings) < num_vectors:
                    embeddings.append(embeddings[-1])
                
                # Truncate if we have too many
                embeddings = embeddings[:num_vectors]
                
                # Return stacked embeddings
                return torch.stack(embeddings)
                
        except Exception as e:
            print(f"Error processing LLM prompts: {e}")
        
        return None
        
    def _read_prompts_from_file(self):
        """Read pre-saved LLM prompts from file."""
        if not (self.sample_name and self.prompt_dir):
            return None

        if self.defect_name:
            prompt_filename = f"{self.sample_name}_{self.defect_name}_prompts.txt"
        else:
            prompt_filename = f"{self.sample_name}_prompts.txt"
        prompt_filepath = os.path.join(self.prompt_dir, prompt_filename)
        
        if not os.path.exists(prompt_filepath):
            print(f"Prompt file not found: {prompt_filepath}")
            return None
        
        try:
            with open(prompt_filepath, 'r') as f:
                content = f.read()
            
            # Extract prompts with regular expressions
            original_prompt_match = re.search(r"Original prompt: (.*?)$", content, re.MULTILINE)
            gen_prompt1_match = re.search(r"Generated prompt 1: (.*?)$", content, re.MULTILINE)
            gen_prompt2_match = re.search(r"Generated prompt 2: (.*?)$", content, re.MULTILINE)
            
            if original_prompt_match and gen_prompt1_match and gen_prompt2_match:
                original_prompt = original_prompt_match.group(1).strip()
                gen_prompt1 = gen_prompt1_match.group(1).strip()
                gen_prompt2 = gen_prompt2_match.group(1).strip()
                
                # Build a response similar to the API return format
                response = (
                    f"Original Image Defect Prompt:\n"
                    f"[{original_prompt}]\n"
                    f"Minor Generalization Image Defect Prompt:\n"
                    f"[{gen_prompt1}]\n"
                    f"[{gen_prompt2}]"
                )
                return response
            else:
                print(f"Error parsing prompt file: {prompt_filepath}")
                
        except Exception as e:
            print(f"Error reading prompt file: {e}")
        
        return None

    def forward(self, tokenized_text, embedded_text):
        """Forward pass: replace placeholder tokens with learned embeddings."""
        b, n, device = *tokenized_text.shape, tokenized_text.device

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)

            if self.max_vectors_per_token == 1:  # Single vector per token: simple replacement
                # Ensure placeholder_token is on the correct device
                placeholder_token = placeholder_token.to(device)
                placeholder_idx = torch.where(tokenized_text == placeholder_token)
                embedded_text[placeholder_idx] = placeholder_embedding
            else:  # Multiple vectors: insert and track shifting indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // 2000  # Adjustable step size
                else:
                    max_step_tokens = self.max_vectors_per_token

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)
                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))

                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat([
                        tokenized_text[row][:col], 
                        placeholder_token.repeat(num_vectors_for_token).to(device), 
                        tokenized_text[row][col + 1:]
                    ], axis=0)[:n]
                    
                    new_embed_row = torch.cat([
                        embedded_text[row][:col], 
                        placeholder_embedding[:num_vectors_for_token], 
                        embedded_text[row][col + 1:]
                    ], axis=0)[:n]

                    embedded_text[row] = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    def llm_regularization_loss(self):
        """Compute LLM regularization loss to keep embeddings close to LLM embeddings."""
        if not self.use_llm_enhancement or not self.llm_embeddings:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        
        loss = 0.0
        count = 0
        
        for key in self.llm_embeddings:
            if key in self.string_to_param_dict:
                optimized = self.string_to_param_dict[key]
                llm_embed = self.llm_embeddings[key].to(optimized.device)
                
                # MSE loss between optimized parameters and LLM embeddings
                loss += F.mse_loss(optimized, llm_embed)
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return self.llm_weight * loss

    def save(self, ckpt_path):
        """Save embedding manager state."""
        save_dict = {
            "string_to_token": self.string_to_token_dict,
            "string_to_param": self.string_to_param_dict
        }
        
        # Save LLM embeddings if present
        if self.llm_embeddings:
            save_dict["llm_embeddings"] = self.llm_embeddings
        
        torch.save(save_dict, ckpt_path)
        print(f"Successfully saved LLM-enhanced embeddings to {ckpt_path}")

    def load(self, ckpt_path):
        """Load embedding manager state."""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]
        
        if "llm_embeddings" in ckpt:
            self.llm_embeddings = ckpt["llm_embeddings"]
        
        print(f"Loaded LLM-enhanced embeddings from {ckpt_path}")

    def embedding_parameters(self):
        """Return parameters that need optimization."""
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):
        """Compute regularization loss toward initial embeddings."""
        loss = 0.
        num_embeddings = len(self.initial_embeddings)
        
        if num_embeddings == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        for key in self.initial_embeddings:
            if key in self.string_to_param_dict:
                optimized = self.string_to_param_dict[key]
                coarse = self.initial_embeddings[key].clone().to(optimized.device)
                
                # MSE loss
                loss = loss + F.mse_loss(optimized, coarse) / num_embeddings

        # Apply original loss weight inversely proportional to LLM weight
        return (1.0 - self.llm_weight) * loss 
    # Dynamic weight support added to existing code
    def _get_regularization_loss(self, current_embeds, llm_embeds, llm_weight=None):
        """
        Compute embedding regularization loss with optional dynamic weight.

        Args:
        - current_embeds: Currently optimized embeddings
        - llm_embeds: LLM-generated embeddings
        - llm_weight: Optional dynamic weight; overrides self.llm_weight when provided
        """
        # Use passed weight or default weight
        weight = llm_weight if llm_weight is not None else self.llm_weight
        
        # Cosine similarity between embeddings
        cosine_sim = F.cosine_similarity(current_embeds, llm_embeds, dim=-1)
        # Convert similarity to distance (1 - cosine_sim)
        distance = 1.0 - cosine_sim
        # Apply weight and return
        return weight * distance.mean()
