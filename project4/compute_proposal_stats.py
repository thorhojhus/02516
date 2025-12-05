"""
Compute proposal statistics for the report.
"""
import json
from pathlib import Path

def compute_stats():
    results_dir = Path("project4/results")

    splits = {
        "train": "train_selective_search_v2.json",
        "val": "val_selective_search_v2.json",
        "test": "test_selective_search_v2.json"
    }

    all_stats = {}

    for split_name, filename in splits.items():
        filepath = Path("project4/processed_data") / filename
        with open(filepath, "r") as f:
            data = json.load(f)

        total_images = len(data)
        total_proposals = 0
        total_positive = 0
        total_negative = 0
        total_gt_boxes = 0

        for entry in data:
            gt_boxes = entry["ground_truths"]
            proposals = entry["labeled_proposals"]

            total_gt_boxes += len(gt_boxes)
            total_proposals += len(proposals)

            for box, label in proposals:
                if label == 1:
                    total_positive += 1
                else:
                    total_negative += 1

        avg_proposals = total_proposals / total_images if total_images > 0 else 0
        avg_gt = total_gt_boxes / total_images if total_images > 0 else 0
        ratio = total_negative / total_positive if total_positive > 0 else float('inf')

        all_stats[split_name] = {
            "total_images": total_images,
            "total_proposals": total_proposals,
            "total_positive": total_positive,
            "total_negative": total_negative,
            "total_gt_boxes": total_gt_boxes,
            "avg_proposals_per_image": avg_proposals,
            "avg_gt_per_image": avg_gt,
            "neg_to_pos_ratio": ratio,
        }

        print(f"\n{'='*60}")
        print(f"{split_name.upper()} SET STATISTICS")
        print(f"{'='*60}")
        print(f"Total images: {total_images}")
        print(f"Total GT boxes: {total_gt_boxes}")
        print(f"Avg GT boxes per image: {avg_gt:.2f}")
        print(f"Total proposals: {total_proposals}")
        print(f"Avg proposals per image: {avg_proposals:.2f}")
        print(f"Total positive proposals: {total_positive}")
        print(f"Total negative proposals: {total_negative}")
        print(f"Negative:Positive ratio: {ratio:.2f}:1")

    # Combined stats
    print(f"\n{'='*60}")
    print("COMBINED STATISTICS")
    print(f"{'='*60}")

    total_images = sum(s["total_images"] for s in all_stats.values())
    total_proposals = sum(s["total_proposals"] for s in all_stats.values())
    total_positive = sum(s["total_positive"] for s in all_stats.values())
    total_negative = sum(s["total_negative"] for s in all_stats.values())
    total_gt = sum(s["total_gt_boxes"] for s in all_stats.values())

    print(f"Total images: {total_images}")
    print(f"Total GT boxes: {total_gt}")
    print(f"Total proposals: {total_proposals}")
    print(f"Total positive: {total_positive}")
    print(f"Total negative: {total_negative}")
    print(f"Overall Neg:Pos ratio: {total_negative/total_positive:.2f}:1")
    print(f"Avg proposals per image: {total_proposals/total_images:.2f}")

    # Generate LaTeX table
    latex = f"""
% Proposal Statistics Table
\\begin{{table}}[h]
\\centering
\\caption{{Region proposal statistics using Selective Search.}}
\\label{{tab:proposals}}
\\begin{{tabular}}{{lccc}}
\\toprule
Metric & Train & Val & Test \\\\
\\midrule
Images & {all_stats['train']['total_images']} & {all_stats['val']['total_images']} & {all_stats['test']['total_images']} \\\\
GT boxes & {all_stats['train']['total_gt_boxes']} & {all_stats['val']['total_gt_boxes']} & {all_stats['test']['total_gt_boxes']} \\\\
Total proposals & {all_stats['train']['total_proposals']} & {all_stats['val']['total_proposals']} & {all_stats['test']['total_proposals']} \\\\
Positive proposals & {all_stats['train']['total_positive']} & {all_stats['val']['total_positive']} & {all_stats['test']['total_positive']} \\\\
Negative proposals & {all_stats['train']['total_negative']} & {all_stats['val']['total_negative']} & {all_stats['test']['total_negative']} \\\\
Avg proposals/image & {all_stats['train']['avg_proposals_per_image']:.1f} & {all_stats['val']['avg_proposals_per_image']:.1f} & {all_stats['test']['avg_proposals_per_image']:.1f} \\\\
Neg:Pos ratio & {all_stats['train']['neg_to_pos_ratio']:.1f}:1 & {all_stats['val']['neg_to_pos_ratio']:.1f}:1 & {all_stats['test']['neg_to_pos_ratio']:.1f}:1 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% Summary statistics
% IoU threshold for positive: 0.5
% IoU threshold for negative: 0.1
% Region proposal method: Selective Search
"""

    with open(results_dir / "proposal_stats.tex", "w") as f:
        f.write(latex)

    print(f"\nSaved proposal statistics to {results_dir / 'proposal_stats.tex'}")

    return all_stats


if __name__ == "__main__":
    compute_stats()
