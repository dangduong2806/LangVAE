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
    
    sentences = [
        "A planet is an astronomical body orbiting a star or stellar remnant.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    print("\n--- Encoding and Decoding Demo ---")
    tokenizer = model.decoder.tokenizer
    
    # Tokenize input sentences using the decoder's tokenizer
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    
    # 1. Encode sentences into the latent space (z)
    print("Encoding sentences into latent space...")
    with torch.no_grad():
        z, cvars_emb = model.encode_z(input_ids)
        
    print(f"Latent representation shape: {z.shape}")
    
    # 2. Decode the latent representations back into sentences
    print("Decoding latent representations...")
    with torch.no_grad():
        reconstructed = model.decode_sentences(z, cvars_emb)
        
    print("\nResults:")
    for orig, recon in zip(sentences, reconstructed):
        print(f"Original     : {orig}")
        print(f"Reconstructed: {recon}")
        print("-" * 50)

if __name__ == "__main__":
    main()
