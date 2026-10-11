"""
Part A — Claude batch OCR pipeline (UNHD Urdu handwriting benchmark)
=====================================================================

Runs your Part A commercial-model benchmark through the REAL Anthropic API
using the Message Batches API (50% cheaper than sync calls, and the token/
cost numbers it reports are real metering, not a proxy's estimate).

Design goals, matching this project's existing methodology:
  - ONE fixed prompt, ONE deterministic decoding policy, same as every other
    Part A model (Section 12's policy) — no per-model prompt tuning.
  - Pre-flight validation before spending anything: confirms every image
    file exists, computes REAL per-image token cost from actual pixel
    dimensions, and prints a total cost estimate you must confirm.
  - custom_id = file_id, so results merge onto evaluation_set.csv the same
    way your other models' predictions CSVs are keyed.
  - Resumable: batch_id is saved to disk so you can close the terminal and
    check back later (batches can take up to 24h, most finish under 1h).
  - Output schema matches your existing *_predictions.csv files (Section
    25's opus_predictions.csv columns) so downstream analysis scripts
    don't need to change.

Usage (run these in order, from the same directory each time):

    python partA_batch_pipeline.py estimate  --eval-csv evaluation_set.csv --images-dir "D:\\Downloads\\gpt 5.5\\archive\\kaggle_upload\\images"
    python partA_batch_pipeline.py submit    --eval-csv evaluation_set.csv --images-dir "D:\\Downloads\\gpt 5.5\\archive\\kaggle_upload\\images"
    python partA_batch_pipeline.py status
    python partA_batch_pipeline.py fetch     --out opus_predictions_real.csv

`estimate` and `submit` both accept --limit N to run a small pilot batch
first (recommended: run a 20-30 image pilot end-to-end before submitting
the full 2,465-image batch).

Requires:  pip install anthropic --break-system-packages
Env var:   ANTHROPIC_API_KEY must be set (real console.anthropic.com key,
           NOT a third-party proxy).
"""

import argparse
import base64
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

try:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
except ImportError:
    sys.exit("Missing dependency. Run: pip install anthropic --break-system-packages")

# ----------------------------------------------------------------------------
# Fixed configuration — change MODEL here if you decide Opus over Sonnet.
# Everything else (prompt, decoding, token cap) stays identical across models,
# per this project's Section 12 policy.
# ----------------------------------------------------------------------------

MODEL = "claude-sonnet-5"          # or e.g. "claude-opus-4-8"

# IMPORTANT: `thinking: {"type": "disabled"}` was tried and did NOT reliably
# take effect in testing -- 292 of 2465 responses still came back with a
# ('thinking',) content block and stop_reason == "max_tokens", producing
# EMPTY prediction text once the 128-token cap was hit mid-thought. Rather
# than keep fighting that setting, the budget is raised generously instead:
# actual transcriptions are short (a few dozen tokens), so this costs
# essentially nothing extra in practice -- it only raises the CEILING, and
# you're billed for tokens actually generated, not the cap.
MAX_OUTPUT_TOKENS = 1024

FIXED_PROMPT = (
    "This image contains a single line of handwritten Urdu text in the Nastaliq "
    "script. Transcribe exactly what is written, character for character, "
    "including any diacritics. Output ONLY the transcribed Urdu text -- no "
    "translation, no explanation, no extra commentary."
)

# Batch API pricing, $ per million tokens, PLUS each model's resolution tier
# (edge px limit, visual-token budget). High-resolution tier = Claude 4.7 and
# later models. This matters for this dataset: most images are wide, thin
# single-line crops that exceed the standard tier's 1568px edge limit and
# get compressed hard on that tier, but mostly fit untouched on high-res.
BATCH_PRICING = {
    "claude-sonnet-5":  {"in": 1.00, "out": 5.00,  "max_edge": 2576, "max_tokens": 4784},
    "claude-opus-5":    {"in": 2.50, "out": 12.50, "max_edge": 2576, "max_tokens": 4784},
    "claude-opus-4-8":  {"in": 2.50, "out": 12.50, "max_edge": 2576, "max_tokens": 4784},
    "claude-opus-4-7":  {"in": 2.50, "out": 12.50, "max_edge": 2576, "max_tokens": 4784},
    "claude-opus-4-6":  {"in": 2.50, "out": 12.50, "max_edge": 1568, "max_tokens": 1568},
    "claude-haiku-4-5": {"in": 0.50, "out": 2.50,  "max_edge": 1568, "max_tokens": 1568},
}

def get_state_file(args) -> Path:
    return Path(getattr(args, "state_file", None) or "batch_state.json")

# The Batch API rejects any single batch whose total request payload exceeds
# 256MB. Base64-encoding images inflates their size by ~4/3, and this dataset's
# full 2,465 images add up to ~537MB base64-encoded -- well over the limit.
# Requests are packed into sub-batches by actual file size (not a fixed count),
# staying safely under the cap, since some images are much larger than others.
MAX_BATCH_PAYLOAD_BYTES = 150_000_000  # 150MB target, safe margin under 256MB


def chunk_rows_by_payload_size(rows, max_bytes=MAX_BATCH_PAYLOAD_BYTES):
    """Greedily group rows into sub-batches so each stays under max_bytes of
    estimated base64+JSON payload."""
    chunks = []
    current = []
    current_bytes = 0
    for r in rows:
        file_bytes = r["image_path"].stat().st_size
        est_bytes = int(file_bytes * 4 / 3) + 600  # base64 expansion + JSON/prompt overhead
        if current and current_bytes + est_bytes > max_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(r)
        current_bytes += est_bytes
    if current:
        chunks.append(current)
    return chunks


# ----------------------------------------------------------------------------
# Image token cost — same ceil(w/28) * ceil(h/28) formula as the Claude docs.
# ----------------------------------------------------------------------------

def count_image_tokens(width: int, height: int) -> int:
    return math.ceil(width / 28) * math.ceil(height / 28)


def resized_size(width: int, height: int, max_edge: int = 1568, max_tokens: int = 1568):
    """Exact port of Anthropic's own resize algorithm. Matters here: most
    images in this dataset are wide, thin single-line crops (up to ~6000px
    wide) that exceed the standard tier's 1568px edge limit -- a naive
    proportional scale-down gives the wrong token count for shapes like
    these, so this binary search (not an approximation) is used."""
    def fits(w, h):
        return (math.ceil(w / 28) * 28 <= max_edge
                and math.ceil(h / 28) * 28 <= max_edge
                and count_image_tokens(w, h) <= max_tokens)
    if fits(width, height):
        return (width, height)
    if height > width:
        rh, rw = resized_size(height, width, max_edge, max_tokens)
        return (rw, rh)
    aspect = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect), 1)):
            lo = mid
        else:
            hi = mid
    return (lo, max(round(lo / aspect), 1))


def image_tokens(width: int, height: int, max_edge: int = 1568, max_tokens: int = 1568) -> int:
    rw, rh = resized_size(width, height, max_edge, max_tokens)
    return count_image_tokens(rw, rh)


def get_png_dimensions(path: Path) -> tuple[int, int]:
    """Read width/height straight from the PNG header. Avoids requiring Pillow
    just to size-check thousands of tiny files."""
    with open(path, "rb") as f:
        data = f.read(26)
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        # Not a PNG (e.g. jpg) -- fall back to Pillow if available.
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_eval_set(eval_csv: Path, images_dir: Path, limit: int | None):
    rows = []
    with open(eval_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["file_id"]
            img_path = images_dir / f"{file_id}.png"
            rows.append({
                "sentence_id": row["sentence_id"],
                "file_id": file_id,
                "writer_id": row["writer_id"],
                "ground_truth": row["text"],
                "image_path": img_path,
            })
    if limit:
        rows = rows[:limit]
    return rows


def validate_images(rows):
    """Pre-flight check: every image must exist before we spend a cent. Prints
    up to 10 missing examples and the total count so a naming mismatch (like
    the kaggle_upload folder possibly using different filenames) gets caught
    here, not after a batch is already submitted."""
    missing = [r for r in rows if not r["image_path"].exists()]
    if missing:
        print(f"\n!! {len(missing)} / {len(rows)} images NOT FOUND at the expected path.")
        for r in missing[:10]:
            print(f"   missing: {r['image_path']}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more.")
        print("\nCheck --images-dir and that filenames match file_id + '.png' exactly.")
        return False
    print(f"All {len(rows)} images found on disk.")
    return True


# ----------------------------------------------------------------------------
# Command: estimate
# ----------------------------------------------------------------------------

def cmd_estimate(args):
    eval_csv = Path(args.eval_csv)
    images_dir = Path(args.images_dir)
    rows = load_eval_set(eval_csv, images_dir, args.limit)

    print(f"Loaded {len(rows)} rows from {eval_csv}")
    if not validate_images(rows):
        sys.exit(1)

    price = BATCH_PRICING.get(MODEL)
    if not price:
        sys.exit(f"No pricing entry for model '{MODEL}' -- add it to BATCH_PRICING.")

    total_in_tokens = 0
    prompt_tokens_est = 60  # rough overhead for the fixed instruction text itself
    sizes = []
    n_downscaled = 0
    for r in rows:
        w, h = get_png_dimensions(r["image_path"])
        sizes.append((w, h))
        if max(w, h) > price["max_edge"]:
            n_downscaled += 1
        total_in_tokens += image_tokens(w, h, price["max_edge"], price["max_tokens"]) + prompt_tokens_est

    est_out_tokens = len(rows) * 40  # typical short transcription; real max cap is 128

    cost_in = total_in_tokens / 1_000_000 * price["in"]
    cost_out = est_out_tokens / 1_000_000 * price["out"]
    worst_case_out_tokens = len(rows) * MAX_OUTPUT_TOKENS
    worst_case_cost = cost_in + worst_case_out_tokens / 1_000_000 * price["out"]

    avg_w = sum(s[0] for s in sizes) / len(sizes)
    avg_h = sum(s[1] for s in sizes) / len(sizes)

    tier = "high-res (2576px/4784tok)" if price["max_edge"] > 1568 else "standard (1568px/1568tok)"
    print(f"\nModel: {MODEL}  [{tier} tier]  (batch pricing: ${price['in']}/MTok in, ${price['out']}/MTok out)")
    print(f"Average image size: {avg_w:.0f}x{avg_h:.0f} px")
    print(f"Images that will be downscaled on this tier: {n_downscaled} / {len(rows)}")
    print(f"Total estimated input tokens:  {total_in_tokens:,}")
    print(f"Estimated output tokens (typical, ~40/image): {est_out_tokens:,}")
    print(f"\nEstimated cost (typical case):  ${cost_in + cost_out:.4f}")
    print(f"Worst-case cost (every response hits the {MAX_OUTPUT_TOKENS}-token cap): ${worst_case_cost:.4f}")
    print(f"\n(For reference, sync/non-batch pricing would be ~2x this.)")


# ----------------------------------------------------------------------------
# Command: submit
# ----------------------------------------------------------------------------

def build_request(row) -> Request:
    with open(row["image_path"], "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return Request(
        custom_id=row["file_id"],
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            # Sonnet 5 runs adaptive thinking ON by default. For a short,
            # deterministic transcription task this both wastes budget and
            # breaks the "one fixed decoding policy across every Part A
            # model" principle -- worse, with a small max_tokens cap the
            # model can spend its entire budget thinking and return NO
            # answer text at all (this is exactly what caused the 271 blank
            # predictions in the first full run). Disable it explicitly.
            thinking={"type": "disabled"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": FIXED_PROMPT},
                ],
            }],
        ),
    )


def cmd_submit(args):
    eval_csv = Path(args.eval_csv)
    images_dir = Path(args.images_dir)
    state_file = get_state_file(args)
    rows = load_eval_set(eval_csv, images_dir, args.limit)

    print(f"Preparing to submit {len(rows)} requests to model '{MODEL}'.")
    if not validate_images(rows):
        sys.exit(1)

    if state_file.exists():
        existing = json.loads(state_file.read_text())
        print(f"\n!! A batch run is already tracked in {state_file}: "
              f"{[b['batch_id'] for b in existing.get('batches', [])]}")
        print("   Delete this file first if you really want to submit a new run.")
        sys.exit(1)

    chunks = chunk_rows_by_payload_size(rows)
    total_payload_mb = sum(r["image_path"].stat().st_size for r in rows) * 4 / 3 / 1e6
    if len(chunks) > 1:
        print(f"\nTotal payload (~{total_payload_mb:.0f} MB base64-encoded) exceeds the Batch "
              f"API's 256MB per-batch limit, so this run will be split into {len(chunks)} "
              f"sub-batches:")
        for i, c in enumerate(chunks, 1):
            print(f"  sub-batch {i}: {len(c)} images")

    if not args.yes:
        confirm = input(f"\nAbout to submit {len(rows)} requests total (as {len(chunks)} "
                         f"sub-batch(es)) to the REAL Anthropic API using model '{MODEL}'. "
                         f"Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    submitted = []
    for i, chunk in enumerate(chunks, 1):
        requests = [build_request(r) for r in chunk]
        batch = client.messages.batches.create(requests=requests)
        print(f"Sub-batch {i}/{len(chunks)} submitted: {batch.id}  ({len(chunk)} requests)")
        submitted.append({"batch_id": batch.id, "n_requests": len(chunk)})

    state_file.write_text(json.dumps({
        "model": MODEL,
        "eval_csv": str(eval_csv),
        "images_dir": str(images_dir),
        "batches": submitted,
    }, indent=2))
    print(f"\nSaved {len(submitted)} batch ID(s) to {state_file} -- use 'status' and 'fetch' to check on them later.")


# ----------------------------------------------------------------------------
# Command: status
# ----------------------------------------------------------------------------

def cmd_status(args):
    state_file = get_state_file(args)
    if not state_file.exists():
        sys.exit(f"No {state_file} found -- submit a batch first.")
    state = json.loads(state_file.read_text())
    client = anthropic.Anthropic()

    all_ended = True
    for i, b in enumerate(state["batches"], 1):
        batch = client.messages.batches.retrieve(b["batch_id"])
        print(f"Sub-batch {i}/{len(state['batches'])}  {batch.id}")
        print(f"  Status: {batch.processing_status}")
        print(f"  Request counts: {batch.request_counts}")
        if getattr(batch, "archived_at", None):
            print(f"  !! Archived at {batch.archived_at} -- results no longer downloadable.")
        elif batch.processing_status != "ended":
            all_ended = False

    if all_ended:
        print("\nAll sub-batches are done. Run 'fetch' to download and score the combined results.")
    else:
        print("\nStill waiting on at least one sub-batch. Most finish within an hour; up to 24h max.")


# ----------------------------------------------------------------------------
# CER / WER scoring (simple Levenshtein-based, no external deps)
# ----------------------------------------------------------------------------

URDU_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")


def strip_diacritics(s: str) -> str:
    return URDU_DIACRITICS.sub("", s)


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def cer(pred, gt):
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gt) / len(gt)


def wer(pred, gt):
    p, g = pred.split(), gt.split()
    if len(g) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return edit_distance(p, g) / len(g)


# ----------------------------------------------------------------------------
# Command: fetch
# ----------------------------------------------------------------------------

def cmd_fetch(args):
    state_file = get_state_file(args)
    if not state_file.exists():
        sys.exit(f"No {state_file} found -- submit a batch first.")
    state = json.loads(state_file.read_text())
    client = anthropic.Anthropic()

    batches = [client.messages.batches.retrieve(b["batch_id"]) for b in state["batches"]]
    not_ended = [b.id for b in batches if b.processing_status != "ended"]
    if not_ended:
        sys.exit(f"{len(not_ended)} sub-batch(es) not finished yet: {not_ended}. Run 'status' to check.")

    # Re-load ground truth so we can score results in the same pass.
    eval_csv = Path(state["eval_csv"])
    gt_by_id = {}
    with open(eval_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_by_id[row["file_id"]] = row

    price = BATCH_PRICING.get(state["model"], {"in": 0, "out": 0})
    total_in, total_out = 0, 0
    out_rows = []

    for b in batches:
        for result in client.messages.batches.results(b.id):
            file_id = result.custom_id
            gt_row = gt_by_id.get(file_id, {})
            gt_text = gt_row.get("text", "")

            if result.result.type == "succeeded":
                msg = result.result.message
                pred_text = "".join(block.text for block in msg.content if block.type == "text").strip()
                in_tok = msg.usage.input_tokens
                out_tok = msg.usage.output_tokens
                total_in += in_tok
                total_out += out_tok
                out_rows.append({
                    "sentence_id": gt_row.get("sentence_id", ""),
                    "file_id": file_id,
                    "writer_id": gt_row.get("writer_id", ""),
                    "ground_truth": gt_text,
                    "prediction": pred_text,
                    "cer_sensitive": cer(pred_text, gt_text),
                    "cer_insensitive": cer(strip_diacritics(pred_text), strip_diacritics(gt_text)),
                    "wer_sensitive": wer(pred_text, gt_text),
                    "wer_insensitive": wer(strip_diacritics(pred_text), strip_diacritics(gt_text)),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "error": "",
                })
            else:
                # errored / expired / canceled -- keep the row, mark clearly, do NOT
                # silently treat as an empty prediction (this is the exact bug this
                # project already caught once in Section 28).
                out_rows.append({
                    "sentence_id": gt_row.get("sentence_id", ""),
                    "file_id": file_id,
                    "writer_id": gt_row.get("writer_id", ""),
                    "ground_truth": gt_text,
                    "prediction": "",
                    "cer_sensitive": "",
                    "cer_insensitive": "",
                    "wer_sensitive": "",
                    "wer_insensitive": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "error": result.result.type,
                })

    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    n_errors = sum(1 for r in out_rows if r["error"])
    real_cost = total_in / 1_000_000 * price["in"] + total_out / 1_000_000 * price["out"]

    print(f"Merged {len(batches)} sub-batch(es) into {out_path} -- {len(out_rows)} rows total")
    print(f"Errors/expired/canceled: {n_errors}")
    print(f"Real token usage -- input: {total_in:,}, output: {total_out:,}")
    print(f"Real batch cost (from actual metered usage): ${real_cost:.4f}")

    scored = [r for r in out_rows if r["cer_sensitive"] != ""]
    if scored:
        mean_cer = sum(r["cer_sensitive"] for r in scored) / len(scored)
        print(f"Mean CER (sensitive), {len(scored)} scored rows: {mean_cer:.4f}")


# ----------------------------------------------------------------------------
# Command: retry-failed -- resubmit ONLY the rows that failed, so you don't
# pay to redo rows that already worked. Two known failure modes so far:
#   1. Empty prediction (was: max_tokens cap hit mid-thought)
#   2. Non-compliant response -- model explains in English instead of just
#      transcribing ("I can see this is... I'm unable to accurately...")
#      on genuinely hard-to-read images, breaking the fixed-prompt policy.
# ----------------------------------------------------------------------------

NONCOMPLIANT_ENGLISH = re.compile(r"[A-Za-z]{4,}")


def is_failed_row(row) -> bool:
    if row["prediction"] == "" and not row["error"]:
        return True
    if NONCOMPLIANT_ENGLISH.search(row["prediction"]):
        return True
    return False


def cmd_retry_failed(args):
    eval_csv = Path(args.eval_csv)
    images_dir = Path(args.images_dir)
    state_file = get_state_file(args)  # defaults to retry_batch_state.json for this command

    with open(args.predictions_csv, newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))
    failed_ids = {r["file_id"] for r in existing_rows if is_failed_row(r)}

    if not failed_ids:
        print(f"No failed rows found in {args.predictions_csv} -- nothing to retry.")
        return

    all_rows = load_eval_set(eval_csv, images_dir, limit=None)
    rows = [r for r in all_rows if r["file_id"] in failed_ids]
    print(f"Found {len(failed_ids)} failed rows in {args.predictions_csv}; "
          f"{len(rows)} of them matched against {eval_csv}.")

    if not validate_images(rows):
        sys.exit(1)

    if state_file.exists():
        existing = json.loads(state_file.read_text())
        print(f"\n!! A retry run is already tracked in {state_file}: "
              f"{[b['batch_id'] for b in existing.get('batches', [])]}")
        print("   Delete this file first if you really want to submit a new retry.")
        sys.exit(1)

    chunks = chunk_rows_by_payload_size(rows)  # almost certainly 1 chunk at this size
    price = BATCH_PRICING[MODEL]
    total_in_tokens = sum(image_tokens(*get_png_dimensions(r["image_path"]),
                                        price["max_edge"], price["max_tokens"]) + 60 for r in rows)
    est_cost = total_in_tokens / 1e6 * price["in"] + len(rows) * 200 / 1e6 * price["out"]
    print(f"Estimated retry cost (rough, with the larger {MAX_OUTPUT_TOKENS}-token budget): ${est_cost:.4f}")

    if not args.yes:
        confirm = input(f"\nAbout to resubmit {len(rows)} previously-failed requests. "
                         f"Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    client = anthropic.Anthropic()
    submitted = []
    for i, chunk in enumerate(chunks, 1):
        requests = [build_request(r) for r in chunk]
        batch = client.messages.batches.create(requests=requests)
        print(f"Retry sub-batch {i}/{len(chunks)} submitted: {batch.id}  ({len(chunk)} requests)")
        submitted.append({"batch_id": batch.id, "n_requests": len(chunk)})

    state_file.write_text(json.dumps({
        "model": MODEL,
        "eval_csv": str(eval_csv),
        "images_dir": str(images_dir),
        "batches": submitted,
    }, indent=2))
    print(f"\nSaved retry batch ID(s) to {state_file}. Check with:")
    print(f"  python {Path(sys.argv[0]).name} status --state-file {state_file}")
    print(f"Then fetch with:")
    print(f"  python {Path(sys.argv[0]).name} fetch --state-file {state_file} --out retry_predictions.csv")


# ----------------------------------------------------------------------------
# Command: merge -- patch retry results back into the original predictions CSV
# ----------------------------------------------------------------------------

def cmd_merge(args):
    with open(args.original, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        original_rows = list(reader)

    with open(args.patch, newline="", encoding="utf-8") as f:
        patch_by_id = {r["file_id"]: r for r in csv.DictReader(f)}

    n_patched = 0
    n_still_failed = 0
    for row in original_rows:
        if row["file_id"] in patch_by_id:
            patch = patch_by_id[row["file_id"]]
            if not is_failed_row(patch):
                row.update(patch)
                n_patched += 1
            else:
                # Retry didn't fix it either -- still patch in the latest attempt
                # (so you see the newest failure, not a stale one) but count it
                # as still-needing-attention rather than silently calling it done.
                row.update(patch)
                n_still_failed += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(original_rows)

    print(f"Wrote merged file to {args.out}")
    print(f"Rows patched from retry: {n_patched}")
    if n_still_failed:
        print(f"!! {n_still_failed} row(s) STILL failed even after retry (empty, or non-compliant "
              f"English response) -- these need a closer look, may be genuinely illegible images.")

    remaining_failed = sum(1 for r in original_rows if is_failed_row(r))
    print(f"Total rows still failed in the merged file: {remaining_failed}")


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in [("estimate", cmd_estimate), ("submit", cmd_submit)]:
        p = sub.add_parser(name)
        p.add_argument("--eval-csv", required=True)
        p.add_argument("--images-dir", required=True)
        p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (pilot run)")
        p.add_argument("--state-file", default="batch_state.json")
        if name == "submit":
            p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
        p.set_defaults(func=fn)

    p = sub.add_parser("status")
    p.add_argument("--state-file", default="batch_state.json")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("fetch")
    p.add_argument("--out", default="predictions_real.csv")
    p.add_argument("--state-file", default="batch_state.json")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("retry-failed")
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--predictions-csv", required=True, help="Existing predictions CSV to scan for empty rows")
    p.add_argument("--state-file", default="retry_batch_state.json")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_retry_failed)

    p = sub.add_parser("merge")
    p.add_argument("--original", required=True, help="Original predictions CSV")
    p.add_argument("--patch", required=True, help="Retry predictions CSV (from 'fetch --state-file retry_batch_state.json')")
    p.add_argument("--out", required=True, help="Where to write the merged, final CSV")
    p.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()