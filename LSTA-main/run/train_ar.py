import os
import sys
import argparse
import time
from typing import List, Optional
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# Ensure root and src are on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fsta.types import CVRPInstance
from data.dataset import (
    L2SegDataset,
    L2SegSample,
    create_training_samples_from_instance,
)
from models.encoder import L2SegEncoder
from models.ar_decoder import L2SegARDecoder


def generate_synthetic_dataset(
    num_instances: int = 25,
    num_customers: int = 50,
    capacity: float = 30.0,
) -> L2SegDataset:
    """Generate synthetic CVRP training samples with alternating DFS sequences."""
    print(f"[*] Generating {num_instances} instances with N={num_customers}, Capacity={capacity}...")
    all_samples: List[L2SegSample] = []

    for i in range(num_instances):
        rng = np.random.RandomState(2000 + i)
        coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, size=(num_customers, 2))])
        demands = np.concatenate([[0.0], rng.randint(1, 8, size=num_customers).astype(np.float64)])
        inst = CVRPInstance(coords=coords, demands=demands, capacity=capacity)

        # Use 4 solver passes to generate richer differing edges and DFS sequences
        samples = create_training_samples_from_instance(inst, solver_passes=4)
        all_samples.extend(samples)

    total_seqs = sum(len(s.ar_sequences) for s in all_samples)
    print(f"[*] Total subproblems: {len(all_samples)} | Total AR sequences: {total_seqs}")
    return L2SegDataset(all_samples)


def train_ar(
    dataset: L2SegDataset,
    epochs: int = 15,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    w_insert: float = 0.8,
    w_delete: float = 0.2,
    save_path: str = "checkpoints/ar_model.pt",
    pretrained_encoder_path: Optional[str] = "checkpoints/nar_model.pt",
    device: str = "cpu",
):
    print(f"\n{'='*75}")
    print(f"         STARTING L2Seg-AR MODEL TRAINING (Sequence Imitation Learning)         ")
    print(f"{'='*75}")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr} | w_insert: {w_insert} | w_delete: {w_delete}")

    # Build models
    encoder = L2SegEncoder(
        node_in_dim=25,
        edge_in_dim=3,
        hidden_dim=hidden_dim,
        num_tfm_layers=2,
        num_gnn_layers=2,
        tfm_heads=2,
        gnn_heads=1,
        ffn_dim=512,
        dropout=0.1,
    ).to(device)

    # Optional warm start: load pretrained encoder from NAR training if available
    if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
        print(f"[*] Loading pretrained encoder weights from: {pretrained_encoder_path}")
        ckpt = torch.load(pretrained_encoder_path, map_location=device, weights_only=False)
        if "encoder_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            print("  [+] Encoder weights loaded successfully as warm start!")

    decoder = L2SegARDecoder(
        hidden_dim=hidden_dim,
        num_delete_layers=1,
        num_insert_layers=4,
        nhead=1,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)

    # Trainable parameters for both Encoder and AR Decoder
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_params, lr=lr)

    # Filter samples that have at least 1 AR sequence
    samples_with_seqs = [s for s in dataset.samples if len(s.ar_sequences) > 0]
    if not samples_with_seqs:
        raise ValueError("No AR sequences found in the dataset! Try generating with more instances or solver passes.")

    split = int(0.85 * len(samples_with_seqs))
    train_samples = samples_with_seqs[:split]
    val_samples = samples_with_seqs[split:]
    if not val_samples:
        val_samples = train_samples[-1:]

    print(f"[*] Train subproblems: {len(train_samples)} | Val subproblems: {len(val_samples)}")

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        encoder.train()
        decoder.train()

        total_train_loss = 0.0
        train_seq_count = 0
        indices = np.random.permutation(len(train_samples))

        for idx in indices:
            s: L2SegSample = train_samples[idx]
            node_feats = s.node_feats.to(device)
            tour_pos = s.tour_positions.to(device)
            route_mem = s.route_membership.to(device)
            edge_index = s.edge_index.to(device)
            edge_attr = s.edge_attr.to(device)

            optimizer.zero_grad()
            # Encode graph
            h_gnn = encoder(node_feats, tour_pos, route_mem, edge_index, edge_attr)

            sub_loss = torch.tensor(0.0, device=device)
            # Accumulate loss over all alternating sequences in this subproblem
            for seq in s.ar_sequences:
                seq_loss = decoder.compute_sequence_loss(
                    h_gnn=h_gnn,
                    node_sequence=seq,
                    local_routes=s.local_routes,
                    w_insert=w_insert,
                    w_delete=w_delete,
                )
                sub_loss = sub_loss + seq_loss
                train_seq_count += 1

            if sub_loss.requires_grad and sub_loss.item() > 0:
                sub_loss.backward()
                clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                total_train_loss += sub_loss.item()

        avg_train_loss = total_train_loss / max(1, train_seq_count)

        # Validation loop
        encoder.eval()
        decoder.eval()
        total_val_loss = 0.0
        val_seq_count = 0

        with torch.no_grad():
            for s in val_samples:
                node_feats = s.node_feats.to(device)
                tour_pos = s.tour_positions.to(device)
                route_mem = s.route_membership.to(device)
                edge_index = s.edge_index.to(device)
                edge_attr = s.edge_attr.to(device)

                h_gnn = encoder(node_feats, tour_pos, route_mem, edge_index, edge_attr)

                for seq in s.ar_sequences:
                    val_seq_loss = decoder.compute_sequence_loss(
                        h_gnn=h_gnn,
                        node_sequence=seq,
                        local_routes=s.local_routes,
                        w_insert=w_insert,
                        w_delete=w_delete,
                    )
                    total_val_loss += val_seq_loss.item()
                    val_seq_count += 1

        avg_val_loss = total_val_loss / max(1, val_seq_count)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss / Seq: {avg_train_loss:.4f} | "
            f"Val Loss / Seq: {avg_val_loss:.4f} | "
            f"Alpha (h_end): {decoder.alpha.item():.3f}"
        )

        # Save checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "val_loss": best_val_loss,
                    "alpha": decoder.alpha.item(),
                },
                save_path,
            )
            print(f"  [+] Saved new best AR weights to: {save_path}")

    print(f"\n[DONE] AR Training complete! Best weights saved at '{save_path}' with Val Loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train L2Seg AR Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--instances", type=int, default=20, help="Number of training instances to generate")
    parser.add_argument("--customers", type=int, default=40, help="Number of customers per instance")
    parser.add_argument("--save_path", type=str, default="checkpoints/ar_model.pt", help="Checkpoint save path")
    parser.add_argument("--pretrained_encoder", type=str, default="checkpoints/nar_model.pt", help="Path to pretrained encoder")
    args = parser.parse_args()

    # Generate synthetic training dataset
    dataset = generate_synthetic_dataset(num_instances=args.instances, num_customers=args.customers)

    # Train AR
    train_ar(
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        pretrained_encoder_path=args.pretrained_encoder,
    )


if __name__ == "__main__":
    main()
