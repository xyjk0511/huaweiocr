# SN barcode validation data

`sn_barcode_manifest.jsonl` is the release-gate manifest for exact barcode-derived SN validation.

Each JSONL row must follow `sn_barcode_manifest.schema.json` and include:

- `image_path`
- `label_id`
- `expected_sn`
- `barcode_present`
- `accepted_quality`
- `notes`

Optional crop paths:

- `sn_path`
- `label_crop`
- `original_image_path`

The validation command defaults to at least 50 accepted-quality barcode-present rows and a 90% exact hit-rate threshold:

```powershell
python validate_sn_barcodes.py --manifest validation/sn_barcode_manifest.jsonl
```

The current repository does not contain enough verified accepted-quality ground-truth rows to claim the 90% release gate. The command intentionally fails when the accepted-quality sample count is below the minimum or when the exact barcode-derived hit rate is below the threshold.
