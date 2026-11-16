import asyncio
from pathlib import Path

from forensics import (
    analyze_image_pipeline,
    clear_account_history,
    ensure_sample_images,
    load_account_history,
    save_to_account_history,
)


async def main():
    base_dir = Path(__file__).resolve().parent
    samples_dir = base_dir / "samples"
    manifest = ensure_sample_images(samples_dir)
    assert len(manifest) == 10, f"Expected 10 forensic sample images, got {len(manifest)}"

    clear_account_history(base_dir, account="mini")

    results = {}
    # Run in reverse so #01 ends up at the top of Mini's history list
    for item in reversed(manifest):
        raw = (samples_dir / item["id"]).read_bytes()
        rep = await analyze_image_pipeline(raw, filename=item["id"])
        save_to_account_history(base_dir, rep, account="mini")
        results[item["id"]] = rep
        assert "visual_layers" in rep and len(rep["visual_layers"]) == 4
        assert 0.0 <= rep["synthetic_probability"] <= 100.0

    for item in manifest:
        rep = results[item["id"]]
        vlm_tag = rep["branch_scores"]["vlm_provider"]
        print(
            f"[OK] {item['title']:<28} | verdict={rep['verdict']:<24} | "
            f"synth={rep['synthetic_probability']:>5.1f}% | vlm={vlm_tag}"
        )

    saved = load_account_history(base_dir, account="mini")
    assert len(saved) == 10, f"Expected 10 reports saved in Mini account, got {len(saved)}"
    print(f"\n[SAVED] {len(saved)} forensic reports persisted to Mini account (data/accounts/mini.json)")


if __name__ == "__main__":
    asyncio.run(main())
