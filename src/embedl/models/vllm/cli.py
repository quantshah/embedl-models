# Copyright (C) 2025 Embedl AB

"""CLI entrypoint for vLLM with FlashHead support"""

import sys
from vllm.entrypoints.cli.main import main


def cli_main():
    """Main entrypoint that ensures FlashHead patching happens before vLLM starts."""
    # Import this to ensure vLLM patching happens
    from embedl.models.vllm import patch_vllm, _create_and_update_model, _load_flash_head_from_checkpoint, _set_flash_head
    
    # Parse args to find the model
    # The model is typically the first positional argument after the subcommand
    model = None
    for i, arg in enumerate(sys.argv):
        if arg in ['serve', 'generate', 'complete']:
            # Next non-flag argument is the model
            for j in range(i + 1, len(sys.argv)):
                if not sys.argv[j].startswith('-'):
                    model = sys.argv[j]
                    break
            break
    
    # Load FlashHead if model is provided
    if model:
        try:
            model_path = _create_and_update_model(model)
            flash_head = _load_flash_head_from_checkpoint(model_path)
            _set_flash_head(flash_head)
            
            # Update the model path in sys.argv
            for i, arg in enumerate(sys.argv):
                if arg == model:
                    sys.argv[i] = model_path
                    break
        except Exception as e:
            # If FlashHead loading fails, continue with standard vLLM
            print(f"[Embedl] Could not load FlashHead: {e}")
            print("[Embedl] Continuing with standard vLLM")
    
    # Run vLLM CLI
    main()


if __name__ == "__main__":
    cli_main()
