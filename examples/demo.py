import torch
from langvae import LangVAE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the pre-trained model from Hugging Face Hub
    model_name = "neuro-symbolic-ai/eb-langvae-bert-base-cased-Qwen2.5-3B-l128"
    print(f"Loading model: {model_name}")
    try:
        model = LangVAE.load_from_hf_hub(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have huggingface_hub installed and are logged in if necessary.")
        return

    # Move components to device
    model.encoder.to(device)
    model.decoder.to(device)
    
    print("\n--- Interactive Encoding and Decoding Demo ---")
    print("Type 'quit' or 'exit' to stop.")
    tokenizer = model.decoder.tokenizer
    
    while True:
        try:
            user_input = input("\nEnter a sentence to reconstruct: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue
                
            sentences = [user_input]
            
            # Tokenize input sentences using the decoder's tokenizer
            tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            
            # 1. Encode sentences into the latent space (z)
            with torch.no_grad():
                z, cvars_emb = model.encode_z(input_ids)
                
            # 2. Decode the latent representations back into sentences
            with torch.no_grad():
                reconstructed = model.decode_sentences(z, cvars_emb)
                
            print(f"\nOriginal     : {sentences[0]}")
            print(f"Reconstructed: {reconstructed[0]}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
