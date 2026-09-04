import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class MultiScaleEmbeddingManager(nn.Module):
    def __init__(
        self,
        tokenizer=None,
        embedding_layer=None,  # Reference to the embedding layer
        placeholder_strings: List[str] = ["*"],
        initializer_words: List[str] = ["defect"],
        per_image_tokens: bool = False,
        num_vectors_per_scale: int = 5,
        scales: int = 3,
        initial_scale_weights: List[float] = [0.4, 0.4, 0.2],
        embedding_dim: int = 768,
        progressive_training: bool = True,
        progressive_steps: List[int] = [0, 500, 1000],
        **kwargs  # Capture extra kwargs
    ):
        """
        Multi-scale embedding manager providing macro, meso, and micro scale
        embeddings for defect representation.

        Args:
            tokenizer: Text tokenizer
            placeholder_strings: List of placeholder strings
            initializer_words: Words used to initialize embeddings
            per_image_tokens: Whether to use different tokens per image
            num_vectors_per_scale: Number of vectors per scale
            scales: Number of scales
            initial_scale_weights: Initial weights for each scale
            embedding_dim: Embedding dimension
            progressive_training: Whether to use progressive training
            progressive_steps: Step thresholds for progressive training
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.placeholder_strings = placeholder_strings
        self.initializer_words = initializer_words
        self.per_image_tokens = per_image_tokens
        self.embedding_dim = embedding_dim
        
        self.progressive_training = progressive_training
        self.progressive_steps = progressive_steps
        self.current_step = 0
        
        # Scale-related parameters
        self.scales = scales
        self.num_vectors_per_scale = num_vectors_per_scale
        self.scale_weights = nn.Parameter(torch.tensor(initial_scale_weights))
        
        # Initialize token and parameter dicts
        self.token_params = nn.ParameterDict()
        self.token_ids = {}
        
        # Create embeddings for each placeholder and scale
        for placeholder_string in placeholder_strings:
            self._init_placeholder_embeddings(placeholder_string)
        
        # Store full token ID mapping for each placeholder
        self.id_replacements = {}
    
    def _init_placeholder_embeddings(self, placeholder_string: str):
        """Initialize multi-scale embeddings for the given placeholder."""
        # Check whether tokenizer is available
        if self.tokenizer is not None:
            base_token_id = self.tokenizer.encode(placeholder_string)[0]
            self.token_ids[placeholder_string] = base_token_id
        else:
            # Temporarily assign an ID; update later
            self.token_ids[placeholder_string] = -1
        base_token_id = self.tokenizer.encode(placeholder_string)[0]
        self.token_ids[placeholder_string] = base_token_id
        
        # Initialize embeddings for each scale
        for scale in range(self.scales):
            scale_key = f"{placeholder_string}_s{scale}"
            
            # Initialize embeddings using initializer words
            init_word = self.initializer_words[0]  # Use the same init word for now
            init_embeddings = self._get_init_embeddings(init_word)
            
            # Create learnable parameters
            self.token_params[scale_key] = nn.Parameter(
                torch.randn(self.num_vectors_per_scale, self.embedding_dim)
            )
            
            # Initialize with embeddings from the initializer word
            with torch.no_grad():
                for i in range(min(self.num_vectors_per_scale, init_embeddings.shape[0])):
                    self.token_params[scale_key][i] = init_embeddings[i]
    
    def _get_init_embeddings(self, init_word: str) -> torch.Tensor:
        """Get embedding representation for the initializer word."""
        # Actual implementation should fetch embeddings from a text encoder (e.g. CLIP)
        # Random initialization is used here as a placeholder
        with torch.no_grad():
            init_ids = self.tokenizer.encode(init_word)
            init_embeddings = torch.randn(len(init_ids), self.embedding_dim)
            # In real code this would be:
            # init_embeddings = self.transformer.token_embedding(torch.tensor(init_ids))
        return init_embeddings
    # Added to MultiScaleEmbeddingManager class
    def set_tokenizer(self, tokenizer):
        """Set tokenizer and update token IDs."""
        self.tokenizer = tokenizer
        self._update_token_ids()

    def _update_token_ids(self):
        """Update token IDs for all placeholders."""
        if self.tokenizer is None:
            return
            
        for placeholder_string in self.placeholder_strings:
            token_id = self.tokenizer.encode(placeholder_string)[0]
            self.token_ids[placeholder_string] = token_id

    # Methods compatible with the original interface
    def get_embedding_norms(self):
        """Return norms of all embeddings."""
        norms = {}
        for placeholder in self.placeholder_strings:
            placeholder_norms = []
            for scale in range(self.scales):
                scale_key = f"{placeholder}_s{scale}"
                if scale_key in self.token_params:
                    norm = torch.norm(self.token_params[scale_key], dim=1).mean().item()
                    placeholder_norms.append(norm)
            norms[placeholder] = placeholder_norms
        return norms
    def get_scale_activation_mask(self) -> List[bool]:
        """Return a mask indicating which scales are active at the current training step."""
        if not self.progressive_training:
            return [True] * self.scales
        
        active_scales = 0
        for step_threshold in self.progressive_steps:
            if self.current_step >= step_threshold:
                active_scales += 1
        
        return [i < active_scales for i in range(self.scales)]
    
    def integrate_scales(self, placeholder_string: str) -> torch.Tensor:
        """Integrate embedding vectors across scales."""
        # Get scale activation mask
        scale_mask = self.get_scale_activation_mask()
        
        # Compute normalized weights
        weights = self.scale_weights.clone()
        # Zero out weights for inactive scales
        for i, active in enumerate(scale_mask):
            if not active:
                weights[i] = 0
        normalized_weights = F.softmax(weights, dim=0)
        
        # Integrate embeddings from each scale
        integrated_embedding = torch.zeros(self.embedding_dim)
        for scale in range(self.scales):
            if scale_mask[scale]:
                scale_key = f"{placeholder_string}_s{scale}"
                scale_embedding = self.token_params[scale_key]
                # Average-pool multiple vectors within each scale
                scale_embedding_pooled = torch.mean(scale_embedding, dim=0)
                integrated_embedding += normalized_weights[scale] * scale_embedding_pooled
        
        return integrated_embedding
    
    def forward(
        self, 
        tokenized_text: torch.Tensor,
        step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Process input text and replace placeholders with learned embeddings.

        Args:
            tokenized_text: Tokenized text tensor [batch_size, seq_len]
            step: Current training step

        Returns:
            modified_embeddings: Modified embedding tensor [batch_size, seq_len, embedding_dim]
        """
        if step is not None:
            self.current_step = step
        
        # Get original embeddings
        # Note: assumes self.transformer.token_embedding exists in actual code
        # In practice, fetch this from the original model
        embedded_text = self.get_embeddings_from_tokens(tokenized_text)
        
        # Process each placeholder
        for placeholder_string in self.placeholder_strings:
            placeholder_id = self.token_ids[placeholder_string]
            
            # Find placeholder positions in the input
            placeholder_mask = (tokenized_text == placeholder_id).to(embedded_text.device)
            
            # If placeholder is found
            if torch.any(placeholder_mask):
                # Get integrated embedding
                integrated_embedding = self.integrate_scales(placeholder_string)
                
                # Replace placeholder with integrated embedding
                # Iterate over each sample in the batch
                for batch_idx in range(embedded_text.shape[0]):
                    for seq_idx in range(embedded_text.shape[1]):
                        if placeholder_mask[batch_idx, seq_idx]:
                            embedded_text[batch_idx, seq_idx] = integrated_embedding
        
        return embedded_text
    # Method to set the embedding layer
    def set_embedding_layer(self, embedding_layer):
        """Set the embedding layer."""
        self.embedding_layer = embedding_layer
    def get_embeddings_from_tokens(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        """Get embeddings from token IDs (placeholder implementation; replace in production)."""
        # Placeholder implementation; should call the original model's embedding layer
        if self.embedding_layer is not None:
            return self.embedding_layer(tokenized_text)
        else:
            # Fallback placeholder implementation
            batch_size, seq_len = tokenized_text.shape
            return torch.randn(batch_size, seq_len, self.embedding_dim).to(tokenized_text.device)
    
    def update_step(self, step: int):
        """Update the current training step."""
        self.current_step = step
    
    def get_active_scales(self) -> List[int]:
        """Return list of currently active scale indices."""
        return [i for i, active in enumerate(self.get_scale_activation_mask()) if active]
