"""
select_images.py

Builds an image dataset from a local folder structure for CLIP training/evaluation.
For each category, scans BASE_PATH/_<name>, applies the min/max rule, selects
images using zone-based sampling, and copies them to OUTPUT_BASE/<name>.

Rules:
    < 2000 images  -> skipped (prompts for a replacement)
    2000-5000      -> all images kept
    > 5000         -> capped at 5000 using proportional zone selection
"""

import os
import sys
import csv
import shutil
import random

# Paths 

BASE_PATH   = "/media/isesat/ae694501-30db-4ab4-a29c-9a6d7ce09450/CT_2023"
OUTPUT_BASE = "/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941"

# Categories 
CATEGORIES: dict[str, list[str]] = {
    "Animals": [
        "ant", "bumblebee", "cow", "elephant","golden_retriever",
        "hamster", "hippo", "hummingbird", "jellyfish", "lion",
        "orca", "penguin","spider","zebra",
    ],
    "Nature & Weather": [
        "astronomy", "autumn", "desert", "fog", "island", "lake", "moon",
        "ocean", "polar_lights", "rainbow", "reef", "snow",
    ],
    "Food & Drink": [
        "apple", "apricot", "beer", "coffee", "egg", "lavender",
        "orange", "pasta", "pineapple", "sushi","wine",
    ],
    "Sports & Activities": [
        "archery", "baseball","parade", "paragliding", "pole_vault", "running",
        "sailing", "soccer", "sumo_wrestling", "surfboard", "wedding","yoga",
    ],
    "Urban & Architecture": [
        "airport", "balcony", "bar", "bathroom", "big_ben", "billboard",
        "bridge", "church", "eiffel_tower", "ferris_wheel",
        "golden_gate_bridge", "harbor", "hospital", "neon_light",
        "petronas_towers", "playground", "windmill",
    ],
    "People & Portraits": [
        "baby",  "portrait_man", "portrait_woman", 
    ],
    "Objects & Technology": [
        "ambulance", "airplane", "balloon", "banknote", "boat", "camera",
        "candle", "chips","coin", "computer_monitor","flag", "hat", "id_document", "infrared",
        "keyboard", "laptop", "microphone", "motorcycle", "power_line",
         "ship", "sign", "smart_phone", "train",
    ],
    "Art & Culture": [
        "christmas","fire", "graffiti", "grave", "organ", "painting", "statue","saxophone",
    ],
}

FOLDER_NAMES: list[str] = [f for names in CATEGORIES.values() for f in names]
FOLDER_CATEGORY: dict[str, str] = {
    f: cat for cat, names in CATEGORIES.items() for f in names
}


# Config 

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
MIN_IMAGES = 2000
MAX_IMAGES = 5000

# Image collection 
def is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def collect_images(folder: str) -> list[str]:
    """Recursively collect all images under folder, sorted by path."""
    images = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for fname in sorted(files):
            if is_image(fname):
                images.append(os.path.join(root, fname))
    return images


def get_source_label(path: str, root: str) -> str:
    parts = os.path.relpath(path, root).split(os.sep)
    return "(root)" if len(parts) == 1 else parts[0]


def group_by_source(images: list[str], root: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in images:
        groups.setdefault(get_source_label(path, root), []).append(path)
    return groups


# Selection 

def proportional_quotas(groups: dict[str, list[str]], n: int) -> dict[str, int]:
    """
    Distribute n selections across subfolders proportionally by size.
    Every subfolder gets at least 1. Uses largest-remainder rounding.
    """
    labels     = sorted(groups.keys())
    total_imgs = sum(len(groups[l]) for l in labels)
    remaining  = n - min(n, len(labels))

    exact      = {l: (len(groups[l]) / total_imgs) * remaining for l in labels}
    floors     = {l: int(exact[l]) + 1 for l in labels}  # +1 for the guarantee
    remainders = {l: exact[l] - int(exact[l]) for l in labels}

    # Distribute leftover slots by largest remainder
    leftover = n - sum(floors.values())
    if leftover > 0:
        for label in sorted(labels, key=lambda l: remainders[l], reverse=True)[:leftover]:
            floors[label] += 1

    # Cap at actual folder size
    for label in labels:
        floors[label] = min(floors[label], len(groups[label]))

    # Top up if capping caused a deficit
    deficit = n - sum(floors.values())
    if deficit > 0:
        for label in sorted(labels, key=lambda l: len(groups[l]) - floors[l], reverse=True):
            add = min(len(groups[label]) - floors[label], deficit)
            floors[label] += add
            deficit -= add
            if deficit == 0:
                break

    return floors


def zone_select(images: list[str], n: int) -> list[str]:
    """Pick n images from a sorted list using equal-width zone sampling."""
    total = len(images)
    if total == 0:
        return []
    num_zones = min(n, total)
    zone_size = total / num_zones
    return [
        random.choice(images[int(z * zone_size):int((z + 1) * zone_size)])
        for z in range(num_zones)
    ]


# File operations 

def copy_images(selected: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    seen: dict[str, int] = {}
    for src in selected:
        base = os.path.basename(src)
        name, ext = os.path.splitext(base)
        if base in seen:
            seen[base] += 1
            dst_name = f"{name}_{seen[base]}{ext}"
        else:
            seen[base] = 0
            dst_name = base
        shutil.copy2(src, os.path.join(output_dir, dst_name))


# Output 
def print_subfolder_breakdown(
    groups: dict[str, list[str]],
    quotas: dict[str, int],
    total_selected: int,
) -> None:
    """Print a per-subfolder table. Skipped if there are no subfolders."""
    labels = sorted(groups.keys())
    if all(l == "(root)" for l in labels):
        return

    total_all = sum(len(groups[l]) for l in labels)
    pct_total = (total_selected / total_all * 100) if total_all else 0

    print(f"    {'Source':<35} {'In folder':>10} {'Selected':>10} {'% taken':>8}")
    print(f"    {'-'*35} {'-'*10} {'-'*10} {'-'*8}")

    ordered = (["(root)"] if "(root)" in labels else []) + [l for l in labels if l != "(root)"]
    for label in ordered:
        total  = len(groups[label])
        sel    = quotas.get(label, 0)
        pct    = (sel / total * 100) if total else 0
        name   = "(loose)" if label == "(root)" else label
        print(f"    {name:<35} {total:>10,} {sel:>10,} {pct:>7.1f}%")

    print(f"    {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
    print(f"    {'total':<35} {total_all:>10,} {total_selected:>10,} {pct_total:>7.1f}%")


def save_csv(records: list[dict], path: str) -> None:
    fields = ["folder", "category", "replaced_from", "found", "saved", "status", "pct_taken"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(records)
    print(f"CSV saved: {path}")


def save_charts(records: list[dict], out_dir: str) -> None:

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed, skipping charts (pip install matplotlib)")
        return

    saved = [r for r in records if r["status"] == "saved"]
    if not saved:
        return

    os.makedirs(out_dir, exist_ok=True)
    category_names = list(CATEGORIES.keys())
    palette     = plt.cm.get_cmap("tab10", len(category_names))
    cat_colours = {cat: palette(i) for i, cat in enumerate(category_names)}

    # Chart 1 — per-folder: found vs saved, sorted by saved count
    sorted_recs = sorted(saved, key=lambda r: r["saved"], reverse=True)
    folders     = [r["folder"] for r in sorted_recs]
    x           = range(len(folders))

    fig, ax = plt.subplots(figsize=(max(16, len(folders) * 0.4), 8))
    ax.bar(x, [r["found"] for r in sorted_recs], color="lightgrey", zorder=2)
    ax.bar(x, [r["saved"] for r in sorted_recs], color=[cat_colours[r["category"]] for r in sorted_recs], zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(folders, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Image count")
    ax.set_title("Images found vs. saved per folder")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(MIN_IMAGES, color="orange", linestyle="--", linewidth=1)
    ax.axhline(MAX_IMAGES, color="red",    linestyle="--", linewidth=1)

    legend_handles = [mpatches.Patch(color=cat_colours[c], label=c) for c in category_names
                      if any(r["category"] == c for r in saved)]
    legend_handles += [
        mpatches.Patch(color="lightgrey", label="found (total)"),
        plt.Line2D([0], [0], color="orange", linestyle="--", label=f"min ({MIN_IMAGES:,})"),
        plt.Line2D([0], [0], color="red",    linestyle="--", label=f"max ({MAX_IMAGES:,})"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    path1 = os.path.join(out_dir, "chart_per_folder.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {path1}")

    # Chart 2 — by category: total saved + folder count
    cat_totals = {c: sum(r["saved"] for r in saved if r["category"] == c) for c in category_names}
    cat_counts = {c: sum(1 for r in saved if r["category"] == c) for c in category_names}
    active     = [c for c in category_names if cat_counts[c] > 0]
    colours    = [cat_colours[c] for c in active]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    bars = ax_left.bar(active, [cat_totals[c] for c in active], color=colours, zorder=2)
    ax_left.set_xticklabels(active, rotation=30, ha="right", fontsize=9)
    ax_left.set_ylabel("Total images saved")
    ax_left.set_title("Total saved images by category")
    ax_left.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_left.set_axisbelow(True)
    for bar, val in zip(bars, [cat_totals[c] for c in active]):
        ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:,}", ha="center", va="bottom", fontsize=8)

    bars = ax_right.bar(active, [cat_counts[c] for c in active], color=colours, zorder=2)
    ax_right.set_xticklabels(active, rotation=30, ha="right", fontsize=9)
    ax_right.set_ylabel("Folders saved")
    ax_right.set_title("Folders saved per category")
    ax_right.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_right.set_axisbelow(True)
    for bar, val in zip(bars, [cat_counts[c] for c in active]):
        ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                      str(val), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Dataset summary by category", fontsize=14)
    fig.tight_layout()
    path2 = os.path.join(out_dir, "chart_by_category.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {path2}")


# Replacement prompt 

def prompt_replacement(original_name: str, original_category: str) -> tuple[str, str, str, list[str], int] | None:
    """
    Ask the user for a replacement folder when the original has too few images.
    Returns (new_name, new_folder_path, chosen_category, images, count) or None to skip.
    """
    cat_list = list(CATEGORIES.keys())

    while True:
        replacement = input(
            f"  replacement for '{original_name}' (or Enter to skip): "
        ).strip()

        if not replacement:
            return None

        repl_path = os.path.join(BASE_PATH, f"_{replacement}")
        if not os.path.isdir(repl_path):
            print(f"  '{repl_path}' not found, try again")
            continue

        images = collect_images(repl_path)
        if len(images) < MIN_IMAGES:
            print(f"  '{replacement}' only has {len(images):,} images, try again")
            continue

        # Valid — now ask for category
        print(f"  '{replacement}' found ({len(images):,} images)")
        print(f"  assign to category:")
        for i, cat in enumerate(cat_list, start=1):
            marker = "  <- original" if cat == original_category else ""
            print(f"    {i}. {cat}{marker}")

        while True:
            choice = input(
                f"  enter number (1-{len(cat_list)}) or Enter to keep [{original_category}]: "
            ).strip()
            if choice == "":
                chosen_category = original_category
                break
            if choice.isdigit() and 1 <= int(choice) <= len(cat_list):
                chosen_category = cat_list[int(choice) - 1]
                break
            print(f"  enter a number between 1 and {len(cat_list)}")

        print(f"  '{replacement}' -> [{chosen_category}]")
        return replacement, repl_path, chosen_category, images, len(images)


# Per-folder processing 

def process_folder(name: str, idx: int, total: int) -> dict:
    input_folder = os.path.join(BASE_PATH, f"_{name}")
    output_path  = os.path.join(OUTPUT_BASE, name)
    category     = FOLDER_CATEGORY.get(name, "Uncategorised")
    original_name = name

    print(f"\n[{idx}/{total}] {name} ({category})")

    record = {
        "folder": name, "category": category, "replaced_from": "",
        "found": 0, "saved": 0, "status": "", "pct_taken": 0.0,
    }

    if not os.path.isdir(input_folder):
        print(f"  folder not found: {input_folder}")
        return {**record, "status": "not found"}

    print(f"  scanning {input_folder}")
    all_images  = collect_images(input_folder)
    total_found = len(all_images)
    record["found"] = total_found

    if total_found < MIN_IMAGES:
        print(f"  {total_found:,} images found, below minimum of {MIN_IMAGES:,}")
        result = prompt_replacement(name, category)
        if result is None:
            print(f"  skipped")
            return {**record, "status": "skipped (too few)"}
        name, input_folder, category, all_images, total_found = result
        output_path = os.path.join(OUTPUT_BASE, name)
        record.update({"folder": name, "category": category,
                        "replaced_from": original_name, "found": total_found})

    if total_found <= MAX_IMAGES:
        num_to_select = total_found
        print(f"  {total_found:,} images, keeping all")
    else:
        num_to_select = MAX_IMAGES
        print(f"  {total_found:,} images, capping at {MAX_IMAGES:,}")

    groups   = group_by_source(all_images, input_folder)
    quotas   = proportional_quotas(groups, num_to_select)
    selected = []
    for label in sorted(groups.keys()):
        selected.extend(zone_select(sorted(groups[label]), quotas[label]))
    random.shuffle(selected)

    print_subfolder_breakdown(groups, quotas, len(selected))

    print(f"  copying {len(selected):,} images to {output_path}")
    copy_images(selected, output_path)

    pct = round(len(selected) / total_found * 100, 1) if total_found else 0
    return {**record, "saved": len(selected), "status": "saved", "pct_taken": pct}


# Main 

def main() -> None:
    print(f"CLIP Dataset Builder")
    print(f"source : {BASE_PATH}/_<name>")
    print(f"output : {OUTPUT_BASE}/<name>")
    print(f"rule   : discard < {MIN_IMAGES:,} | keep <= {MAX_IMAGES:,} | cap at {MAX_IMAGES:,}")
    print(f"folders: {len(FOLDER_NAMES)}")

    records = []
    for idx, name in enumerate(FOLDER_NAMES, start=1):
        records.append(process_folder(name, idx, len(FOLDER_NAMES)))

    saved   = [r for r in records if r["status"] == "saved"]
    skipped = [r for r in records if r["status"] != "saved"]

    print(f"\n--- results ({len(saved)}/{len(FOLDER_NAMES)} saved) ---")
    print(f"\n  {'folder':<30} {'category':<25} {'found':>8} {'saved':>8} {'%':>7}")
    print(f"  {'-'*30} {'-'*25} {'-'*8} {'-'*8} {'-'*7}")
    for r in saved:
        note = f"  (replaces {r['replaced_from']})" if r.get("replaced_from") else ""
        print(f"  {r['folder']:<30} {r['category']:<25} {r['found']:>8,} {r['saved']:>8,} {r['pct_taken']:>6.1f}%{note}")
    if skipped:
        print(f"\n  skipped ({len(skipped)}):")
        for r in skipped:
            print(f"    {r['folder']:<30} {r['status']}")

    csv_path = os.path.join(OUTPUT_BASE, "dataset_summary.csv")
    save_csv(records, csv_path)
    save_charts(records, OUTPUT_BASE)


if __name__ == "__main__":
    main()