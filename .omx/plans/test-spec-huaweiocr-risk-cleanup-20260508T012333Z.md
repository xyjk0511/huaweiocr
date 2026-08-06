# Test Spec: HuaweiOCR Risk Cleanup

## Required Checks
1. Unit regression: `python -m unittest tests.test_sn_barcode_scanning`
2. Full repository tests: `python -m unittest discover -s tests`
3. Diff review for source policy:
   - No production or validation path appends `("original", ...)` for SN scanning.
   - OpenSpec no longer advertises original-image fallback as active.

## Evidence Expectations
- Test commands must be freshly run after edits.
- Full-suite failures must be fixed or explicitly explained as unrelated blockers.
- Review must include file/line references for remaining risk if any.
