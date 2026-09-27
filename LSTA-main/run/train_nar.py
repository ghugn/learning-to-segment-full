import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim

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
from models.nar_decoder import L2SegNARDecoder


def generate_synthetic_dataset(num_instances: int = 20, num_customers: int = 50, capacity: float = 30.0) -> L2SegDataset:
    """Generate synthetic training samples using local search oracle."""
    print(f"[*] Generating {num_instances} instances with N={num_customers}, Capacity={capacity}...")
    all_samples: list[L2SegSample] = []
    
    for i in range(num_instances):
        rng = np.random.RandomState(1000 + i)
        coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, size=(num_customers, 2))])
        demands = np.concatenate([[0.0], rng.randint(1, 8, size=num_customers).astype(np.float64)])
        inst = CVRPInstance(coords=coords, demands=demands, capacity=capacity)
        
        samples = create_training_samples_from_instance(inst, solver_passes=3)
        all_samples.extend(samples)

    print(f"[*] Total subproblem training samples collected: {len(all_samples)}")
    return L2SegDataset(all_samples)


def train_nar(
    dataset: L2SegDataset,
    epochs: int = 15,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    w_pos: float = 9.0,
    save_path: str = "checkpoints/nar_model.pt",
    device: str = "cpu",
):
    print(f"\n{'='*70}")
    print(f"         STARTING L2Seg-NAR MODEL TRAINING (Imitation Learning)         ")
    print(f"{'='*70}")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr} | Pos Weight (w_pos): {w_pos}")

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

    decoder = L2SegNARDecoder(hidden_dim=hidden_dim, dropout=0.1).to(device)

    # Optimizer updates both Encoder and Decoder weights
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_params, lr=lr)

    num_samples = len(dataset)
    split = int(0.85 * num_samples)
    train_samples = dataset.samples[:split]
    val_samples = dataset.samples[split:]

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        encoder.train()
        decoder.train()

        total_train_loss = 0.0
        # Shuffle training samples
        indices = np.random.permutation(len(train_samples))

        for idx in indices:
            s: L2SegSample = train_samples[idx]
            node_feats = s.node_feats.to(device)
            tour_pos = s.tour_positions.to(device)
            route_mem = s.route_membership.to(device)
            edge_index = s.edge_index.to(device)
            edge_attr = s.edge_attr.to(device)
            labels = s.nar_labels.to(device)

            optimizer.zero_grad()
            h_gnn = encoder(node_feats, tour_pos, route_mem, edge_index, edge_attr)
            p_nar = decoder(h_gnn)

            loss = decoder.compute_loss(p_nar, labels, w_pos=w_pos)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_samples))

        # Validation loop
        encoder.eval()
        decoder.eval()
        total_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for s in val_samples:
                node_feats = s.node_feats.to(device)
                tour_pos = s.tour_positions.to(device)
                route_mem = s.route_membership.to(device)
                edge_index = s.edge_index.to(device)
                edge_attr = s.edge_attr.to(device)
                labels = s.nar_labels.to(device)

                h_gnn = encoder(node_feats, tour_pos, route_mem, edge_index, edge_attr)
                p_nar = decoder(h_gnn)

                val_loss = decoder.compute_loss(p_nar, labels, w_pos=w_pos)
                total_val_loss += val_loss.item()

                all_preds.extend((p_nar >= 0.5).cpu().numpy().astype(int))
                all_targets.extend(labels.cpu().numpy().astype(int))

        avg_val_loss = total_val_loss / max(1, len(val_samples))

        # Compute precision & recall on unstable (positive) nodes
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        true_pos = np.sum((preds_arr == 1) & (targets_arr == 1))
        pred_pos = np.sum(preds_arr == 1)
        actual_pos = np.sum(targets_arr == 1)

        precision = true_pos / max(1, pred_pos)
        recall = true_pos / max(1, actual_pos)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Precision: {precision:.3f} | Recall: {recall:.3f}"
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
                    "precision": precision,
                    "recall": recall,
                },
                save_path,
            )
            print(f"  [+] Saved new best model weights to: {save_path}")

    print(f"\n[DONE] Training complete! Best weights saved at '{save_path}' with Val Loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train L2Seg NAR Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--instances", type=int, default=15, help="Number of training instances to generate")
    parser.add_argument("--customers", type=int, default=40, help="Number of customers per instance")
    parser.add_argument("--save_path", type=str, default="checkpoints/nar_model.pt", help="Checkpoint save path")
    args = parser.parse_args()

    # Generate synthetic training dataset
    dataset = generate_synthetic_dataset(num_instances=args.instances, num_customers=args.customers)

    # Train NAR
    train_nar(
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
