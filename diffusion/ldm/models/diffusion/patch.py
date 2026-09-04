def patch_embedding_manager(model):
    """Set required dependencies on the embedding manager after model initialization."""
    if hasattr(model, 'embedding_manager'):
        # Set tokenizer
        if hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'tokenizer'):
            model.embedding_manager.set_tokenizer(model.cond_stage_model.tokenizer)
        
        # Set embedding_layer - access embedding layer according to CLIP model structure
        if hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'transformer'):
            # Try multiple possible paths to obtain the embedding layer
            embedding_layer = None
            transformer = model.cond_stage_model.transformer
            
            # Method 1: direct access
            if hasattr(transformer, 'token_embedding'):
                embedding_layer = transformer.token_embedding
            
            # Method 2: access via text_model
            elif hasattr(transformer, 'text_model') and hasattr(transformer.text_model, 'embeddings'):
                if hasattr(transformer.text_model.embeddings, 'token_embedding'):
                    embedding_layer = transformer.text_model.embeddings.token_embedding
                elif hasattr(transformer.text_model.embeddings, 'word_embeddings'):
                    embedding_layer = transformer.text_model.embeddings.word_embeddings
            
            # Method 3: embeddings attribute
            elif hasattr(transformer, 'embeddings'):
                if hasattr(transformer.embeddings, 'token_embedding'):
                    embedding_layer = transformer.embeddings.token_embedding
                elif hasattr(transformer.embeddings, 'word_embeddings'):
                    embedding_layer = transformer.embeddings.word_embeddings
            
            # Set embedding layer if found
            if embedding_layer is not None:
                model.embedding_manager.set_embedding_layer(embedding_layer)
            else:
                print("Warning: Could not find embedding layer in CLIP model")
    
    return model
