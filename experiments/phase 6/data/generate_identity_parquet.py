import csv
import random
import os
import sys
import pandas as pd

# Add project root to sys.path to import project tokenizer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.utils.tokenizer import get_tokenizer

# Paths
CSV_PATH = os.path.join(os.path.dirname(__file__), "identity_qa.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "data")
OUT_FILE = os.path.join(OUT_DIR, "data_identity.parquet")
OLD_FILE = os.path.join(OUT_DIR, "identity_750.parquet")

# 10 paraphrased questions for each of the 15 original questions
PARAPHRASES = {
    0: [
        "Can you introduce yourself and describe your fundamental architecture?",
        "What model are you, and what is the underlying architecture behind your design?",
        "Could you provide an overview of your identity and core neural architecture?",
        "What are you called, and what kind of model architecture drives your system?",
        "Who created you and what is the fundamental architecture of BEBLaDII?",
        "What type of artificial intelligence are you in terms of model architecture?",
        "Explain your identity and the architectural foundation you operate on.",
        "What is your name and which model family or architecture do you belong to?",
        "Can you describe what you are and how your architecture is fundamentally constructed?",
        "Introduce yourself: what is your name and what architecture do you use?"
    ],
    1: [
        "In what fundamental ways does your reasoning process differ from autoregressive LLMs like GPT?",
        "How does your generation mechanism contrast with traditional next-token prediction models?",
        "Why isn't your thinking process autoregressive like GPT or LLaMA?",
        "What separates your generation process from classic token-by-token language models?",
        "How does BEBLaDII's reasoning paradigm differ from the GPT family of models?",
        "Could you contrast your generation technique with next-word prediction autoregression?",
        "Why do you not generate responses token-by-token like GPT?",
        "What makes your thinking and generation distinct from causal autoregressive architectures?",
        "How does your diffusion-based generation differ fundamentally from autoregressive language models?",
        "Can you explain how your inference process diverges from standard GPT-like next-token generation?"
    ],
    2: [
        "Can you explain how your system is structured to process information?",
        "What are System 1 and System 2 in your architecture, and how do they interact?",
        "How do you organize the flow of data across your discrete and continuous components?",
        "Could you break down the structural layout of your information processing pipeline?",
        "What is the relationship between your verbal space and your latent thinking core?",
        "Describe the division of labor between your System 1 and System 2.",
        "How does your architecture process inputs from raw text to continuous vectors?",
        "What comprises the overall information processing mechanism in BEBLaDII?",
        "How do discrete tokens and continuous representations coexist in your framework?",
        "Could you detail the overall architectural structure of your processing pipeline?"
    ],
    3: [
        "How do you comprehend and condition on user prompts without using causal attention?",
        "Since you are non-autoregressive, how do user inputs steer your generation?",
        "How does a user's prompt influence your output during diffusion?",
        "In what way is the user query ingested and utilized during response denoising?",
        "How does the model incorporate the question into the answer canvas without concatenating tokens?",
        "Through what mechanism do you process and condition upon user queries?",
        "How does your network stay aligned with the user prompt during generation?",
        "Can you explain how user questions are ingested via Latent Encoder and CA_Prompt?",
        "How do you attend to the prompt throughout the denoising trajectory?",
        "What ensures that your generated canvas directly answers the user's specific question?"
    ],
    4: [
        "What neural network serves as your latent backbone, and how was it developed?",
        "Could you describe the structure of latentBERT and how DUS was applied to ModernBERT?",
        "What is the backbone architecture powering your latent diffusion core?",
        "How does Depth Up-Scaling (DUS) factor into your 40-layer transformer backbone?",
        "Which pre-trained model was used as the foundation for your diffusion backbone?",
        "Tell me about the technical details of your latent transformer layers.",
        "How was your 40-layer latent backbone constructed from ModernBERT-large?",
        "What makes up the underlying neural backbone of your diffusion model?",
        "Could you elaborate on the architecture of your latent core and its layer expansion?",
        "What is the internal design of latentBERT and how does it process latent vectors?"
    ],
    5: [
        "How does the model determine its internal confidence while forming an answer?",
        "What mechanism allows you to evaluate token uncertainty during generation?",
        "How does the Sensor Ensemble gauge whether generated tokens are reliable?",
        "In what way do you monitor token confidence and semantic conflict during denoising?",
        "What indicators or metrics do you use to measure your own confidence at each step?",
        "How do you prevent hallucinations or false confidence during response formation?",
        "Could you explain how Sensor Ensemble assesses the state of intermediate vectors?",
        "By what method do you track confidence across the denoising trajectory?",
        "How do metrics like RawDProx and ConflictSim inform your confidence estimation?",
        "What role does the Sensor Ensemble play in detecting uncertainty in your representations?"
    ],
    6: [
        "What is the role of the Orchestrator in BEBLaDII and how does it operate?",
        "Can you describe the responsibilities and algorithms of the Orchestrator module?",
        "How does the Orchestrator guide the fate of individual tokens on the canvas?",
        "What makes the Orchestrator distinct from the neural network itself?",
        "Could you explain the algorithmic decision-making carried out by the Orchestrator?",
        "How does the Orchestrator interact with Sensor Ensemble signals to control diffusion?",
        "What actions can the Orchestrator take when managing tokens during inference?",
        "Why is the Orchestrator implemented algorithmically rather than as a neural head?",
        "In what ways does the Orchestrator manage token freezing, continuation, and memory calls?",
        "Explain how the Orchestrator governs the lifecycle of generated tokens."
    ],
    7: [
        "How does your model store and retrieve factual knowledge without relying on huge parametric memory?",
        "What is Complementary Latent Memory (CLM) and how does it function?",
        "How do you access factual information when encountering knowledge limits?",
        "Describe your approach to external knowledge retrieval and latent memory injection.",
        "How is RAG integrated into your continuous latent space via CLM?",
        "Why don't you rely on memorizing billions of facts directly in your weights?",
        "In what manner does the model query and incorporate external factual knowledge?",
        "How does CLM inject precise facts without converting text back and forth?",
        "What happens when your model reaches a semantic dead end regarding a factual query?",
        "Explain the mechanism of latent fact injection in BEBLaDII."
    ],
    8: [
        "What occurs if the response canvas has more tokens than needed for a concise answer?",
        "How does the model handle unused space on the answer canvas?",
        "What is the function of the <|void|> token when generating short responses?",
        "How does BEBLaDII represent semantic emptiness on an oversized canvas?",
        "Why don't extra canvas tokens degrade or pad-pollute your concise answers?",
        "How does the Orchestrator treat <|void|> tokens upon completion of generation?",
        "Can you explain how short answers are accommodated on wide diffusion canvases?",
        "What mechanism prevents the model from generating unnecessary filler tokens?",
        "How does the <|void|> token maintain spherical geometry while indicating empty space?",
        "Describe how your system compacts answers when given an overly generous canvas."
    ],
    9: [
        "How do you handle complex concepts that cannot fit within the initial canvas length?",
        "What is the purpose and mechanics of the <|expand|> token?",
        "How does the canvas dynamically resize when an answer requires more room?",
        "What does the Orchestrator do when it encounters an <|expand|> token?",
        "In what way does the model unpack compressed semantic representations into new canvas space?",
        "How does dynamic canvas expansion work during complex reasoning steps?",
        "What happens when an idea is too dense to be fully unpacked in the current sequence?",
        "Explain the process of pausing diffusion to insert spherical noise for canvas expansion.",
        "How does the system ensure long, multi-faceted reasoning can unpack beyond fixed windows?",
        "Describe the role of <|expand|> in scaling answer capacity dynamically."
    ],
    10: [
        "Why are all internal representations in BEBLaDII normalized onto a hypersphere?",
        "What are the advantages of using a spherical latent space instead of unbounded Euclidean space?",
        "How does spherical geometry stabilize your deep 40-layer diffusion process?",
        "What does it mean that semantic meaning is defined by angles rather than magnitudes?",
        "Why is unit-norm normalization (L2=1) critical for latent diffusion in your architecture?",
        "How does the hypersphere formulation prevent variance collapse and gradient explosions?",
        "Could you explain the mathematical motivation behind your spherical latent space?",
        "How does operating on a unit hypersphere affect distance and similarity calculations?",
        "What role does directional angular geometry play in your latent representations?",
        "Why does BEBLaDII enforce hyperspherical constraints across all latent states?"
    ],
    11: [
        "How exactly do the CA_Prompt layers connect the user prompt to the canvas?",
        "What is the attention mechanism in CA_Prompt and why is it global?",
        "How do queries, keys, and values interact between the prompt and the answer canvas?",
        "Why does each canvas token attend to the entire question in CA_Prompt?",
        "Could you detail how cross-attention injects prompt semantics into diffusion layers?",
        "How does CA_Prompt ensure that the denoising trajectory remains steered by the prompt?",
        "What is the mathematical and architectural structure of a CAPromptLayer?",
        "Why is global full attention used rather than local causal masking in CA_Prompt?",
        "How do the prompt embeddings serve as keys and values for canvas queries?",
        "Explain how cross-attention prevents canvas tokens from diverging from the question."
    ],
    12: [
        "Are you designed for factual knowledge retrieval or logical reasoning?",
        "How does your focus on reasoning distinguish you from traditional fact-memorizing LLMs?",
        "Why is BEBLaDII optimized for logic rather than encyclopedic knowledge?",
        "How do you approach questions requiring logical deduction versus factual recall?",
        "What are the limitations of your knowledge base, and how are they intentional?",
        "Can you compare your reasoning capabilities with memorized encyclopedic lookup?",
        "Why does your architecture prioritize structured thinking over storing massive facts?",
        "How does the model handle tasks where deductive reasoning is paramount?",
        "Why shouldn't users expect you to know obscure trivia without CLM support?",
        "What makes your architecture specifically suited for reasoning rather than fact regurgitation?"
    ],
    13: [
        "Why is it important for you to possess explicit self-knowledge of your architecture?",
        "How does understanding your own architecture serve as an anchor for alignment?",
        "Why not treat knowledge of your own nature as external information to be retrieved?",
        "In what way does architectural self-awareness help prevent hallucinations about your capabilities?",
        "How does knowing your Latent Diffusion foundation guide your behavioral boundaries?",
        "Why is model identity considered an intrinsic foundation rather than trivial metadata?",
        "How does identity alignment prevent the model from assuming it works like autoregressive models?",
        "What purpose does internalizing your system's constraints serve during inference?",
        "Why is architectural grounding necessary for consistent reasoning about your own limits?",
        "How does knowing you are BEBLaDII help maintain system stability?"
    ],
    14: [
        "Can you explain the distinction between t_local and t_global during inference?",
        "What is hierarchical noise scheduling and how do t_global and t_local interact?",
        "Why do individual tokens have independent noise levels (t_local) instead of a single global step?",
        "How does per-token t_local enable asynchronous crystallization of parts of the answer?",
        "What does t_global represent compared to the actual noise level of an individual token?",
        "How does having local noise levels prevent the model from being bound to rigid lockstep denoising?",
        "Could you explain how t_actual/t_local allows easy tokens to freeze while hard ones continue diffusing?",
        "What role does the Orchestrator play in updating t_local for distinct canvas positions?",
        "Why is hierarchical noise scheduling superior to uniform global timesteps across all tokens?",
        "How does the interaction of macro-level t_global and micro-level t_local shape response formation?"
    ]
}

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Source CSV not found: {CSV_PATH}")

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} base QA pairs from {CSV_PATH}")

    tokenizer = get_tokenizer()
    print("Tokenizer loaded successfully.")

    # Build unique pool: 15 original + 150 paraphrased = 165 pairs
    unique_pool = []
    for idx, row in enumerate(rows):
        orig_q = row["question"]
        ans = row["answer"]
        tokens = tokenizer.encode(ans, add_special_tokens=False)
        length = len(tokens)

        unique_pool.append({"Q": orig_q, "A": ans, "length": length})
        
        paraphrases = PARAPHRASES.get(idx, [])
        for p_q in paraphrases:
            unique_pool.append({"Q": p_q, "A": ans, "length": length})

    print(f"Constructed unique pool of {len(unique_pool)} QA pairs.")

    # Target: 750 samples
    target_count = 750
    rng = random.Random(42)

    full_samples = []
    repeats = target_count // len(unique_pool)
    for _ in range(repeats):
        full_samples.extend(unique_pool.copy())
    
    remaining = target_count - len(full_samples)
    if remaining > 0:
        full_samples.extend(rng.sample(unique_pool, remaining))

    rng.shuffle(full_samples)

    df = pd.DataFrame(full_samples)
    # Ensure canonical column order
    df = df[["Q", "A", "length"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved {len(df)} samples with schema {df.columns.tolist()} to:\n{OUT_FILE}")

    # Remove deprecated file if exists
    if os.path.exists(OLD_FILE):
        os.remove(OLD_FILE)
        print(f"Removed deprecated file: {OLD_FILE}")

if __name__ == "__main__":
    main()
