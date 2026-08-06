## ADDED Requirements

### Requirement: Barcode-first SN extraction
The system SHALL attempt barcode decoding for SN extraction before any OCR-based SN extraction.

#### Scenario: Barcode SN is decoded
- **WHEN** barcode decoding returns exactly one parseable SN candidate for a label
- **THEN** the system SHALL output that SN with `sn_src` set to `barcode`

#### Scenario: Barcode fails before OCR fallback
- **WHEN** no barcode decoder returns a parseable SN candidate
- **THEN** the system MAY run OCR fallback, but the fallback result SHALL NOT be counted as a barcode hit

### Requirement: Multi-source SN barcode scanning
The system SHALL scan SN barcodes from all available local visual sources for a label, including the SN crop when present, the full label crop, and generated barcode-region candidates.

#### Scenario: Original source photo is available
- **WHEN** the manifest preserves an original source image path for provenance
- **THEN** the system SHALL NOT scan that full source image as an SN barcode source for a cropped label

#### Scenario: SN crop is missing
- **WHEN** the field detector does not produce an SN crop but a label crop exists
- **THEN** the system SHALL still attempt SN barcode decoding on the label crop and generated barcode-region candidates

#### Scenario: Multiple barcode sources are available
- **WHEN** an SN crop and a label crop both exist
- **THEN** the system SHALL aggregate barcode candidates from both sources before deciding whether to use OCR

### Requirement: SN barcode candidate selection
The system SHALL only accept barcode payloads that parse into a valid SN according to SN extraction rules.

#### Scenario: Non-SN barcode is decoded
- **WHEN** barcode decoding returns logistics, EAN, QR, or other non-SN payloads
- **THEN** the system SHALL keep those payloads as diagnostics and SHALL NOT output them as `sn`

#### Scenario: Ambiguous barcode SN candidates
- **WHEN** barcode decoding returns conflicting parseable SN candidates for the same label
- **THEN** the system SHALL reject automatic barcode selection for that label and report the ambiguity in diagnostics

### Requirement: Barcode hit-rate validation
The system SHALL provide a deterministic validation command that measures exact SN barcode hit rate against a ground-truth dataset.

#### Scenario: Validation reaches threshold
- **WHEN** the validation dataset contains accepted-quality SN barcode samples and the exact barcode-derived SN hit rate is at least 90%
- **THEN** the validation command SHALL pass and report the numerator, denominator, and hit rate

#### Scenario: Validation misses threshold
- **WHEN** the exact barcode-derived SN hit rate is below 90%
- **THEN** the validation command SHALL fail and list failure counts by decoder miss, SN parse failure, ambiguous barcode, and quality rejection

### Requirement: OCR fallback accounting
The system SHALL report OCR recovery separately from barcode hit rate.

#### Scenario: OCR recovers an SN after barcode miss
- **WHEN** barcode decoding fails and OCR extracts an SN
- **THEN** the system SHALL output the OCR-derived SN with an OCR source and SHALL count the row as OCR recovery, not barcode success

### Requirement: Image quality diagnostics
The system SHALL report when an SN barcode cannot be fairly evaluated because the image is below accepted barcode quality.

#### Scenario: Barcode is too small or unreadable
- **WHEN** the visible SN barcode is clipped, severely blurred, missing quiet zones, or below the configured minimum barcode-pixel threshold
- **THEN** the system SHALL mark the row as a quality reject and SHALL include it in quality diagnostics

### Requirement: Barcode metrics in pipeline output
The system SHALL expose barcode-specific aggregate metrics for each pipeline run.

#### Scenario: Pipeline completes
- **WHEN** the pipeline finishes processing input images
- **THEN** it SHALL report SN barcode attempts, barcode hits, barcode hit rate, OCR recoveries, barcode parse failures, decoder misses, ambiguous barcode cases, and quality rejects
