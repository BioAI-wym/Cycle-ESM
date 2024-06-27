import evo_prot_grad
from transformers import AutoTokenizer, EsmForMaskedLM

def process_sequence(raw_protein_sequence):
    # Convert raw protein sequence to FASTA format
    fasta_format_sequence = f">Input_Sequence\n{raw_protein_sequence}"
    temp_fasta_path = "temp_input_sequence.fasta"
    with open(temp_fasta_path, "w") as file:
        file.write(fasta_format_sequence)

    # Load the ESM-2 model and tokenizer
    esm2_expert = evo_prot_grad.get_expert(
        'esm',
        model=EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D"),
        tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"),
        temperature=0.95,
        device='cuda'  # Use 'cpu' if GPU is not available
    )

    # Initialize Directed Evolution with ESM-2
    directed_evolution = evo_prot_grad.DirectedEvolution(
        wt_fasta=temp_fasta_path,
        output='best',
        experts=[esm2_expert],
        parallel_chains=1,
        n_steps=20,
        max_mutations=10,
        verbose=True
    )

    # Run the evolution process
    variants, scores = directed_evolution()

    # Return the best variant (or customize as needed)
    best_variant = max(zip(variants, scores), key=lambda x: x[1])[0]
    return best_variant

def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        sequences = file.readlines()

    processed_sequences = [process_sequence(seq.strip()) for seq in sequences]

    with open(output_file, 'w') as file:
        for seq in processed_sequences:
            file.write(seq + '\n')

# Process positive and negative sequences
process_file('Data/3/main/neg.txt', 'Data/3/mod3/neg.txt')
process_file('Data/3/main/pos.txt', 'Data/3/mod3/pos.txt')
