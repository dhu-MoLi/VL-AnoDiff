from pytorch_lightning.callbacks import Callback

class MultiScaleTrainingCallback(Callback):
    """Multi-scale training callback that controls scale activation during training."""
    
    def __init__(self, progressive_steps=[0, 500, 1000]):
        super().__init__()
        self.progressive_steps = progressive_steps
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Actions at the start of each training batch."""
        try:
            # Try to update current step in the embedding manager
            if hasattr(pl_module, 'embedding_manager'):
                pl_module.embedding_manager.update_step(trainer.global_step)
            elif hasattr(pl_module, 'cond_stage_model') and hasattr(pl_module.cond_stage_model, 'embedding_manager'):
                pl_module.cond_stage_model.embedding_manager.update_step(trainer.global_step)
        except Exception as e:
            print(f"Warning in MultiScaleTrainingCallback: {e}")
