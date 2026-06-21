# Project Concepts

## OCR Pipeline

### Stage1 Label Crop

A crop that isolates one physical Huawei device label from a source photo before field-level extraction begins.

### Stage2 Field Crop

A label-local crop that isolates one named field from a Stage1 label crop for recognition, review, and metric reporting.

### PartNo Field

The Huawei label field made of visible `Part No.:` text, the printed part-number value, and the paired part-number barcode.

PartNo is treated as barcode-backed for every label in this project, so a valid PartNo crop must preserve the text/value band and the corresponding barcode rather than only one of them.

### Model Field

The Huawei label field that carries the device model value, usually near a model barcode and optional description text.

Model recognition may use PartNo evidence from the same label when the model field itself is incomplete or when the model barcode is not the first barcode decoded.

### SN Field

The Huawei label field made of visible `S/N:` text, the printed serial value, and the paired serial-number barcode.

The SN Field is the primary source for serial recognition, and its barcode result is counted separately from any OCR fallback.

### Barcode-First Recognition

The recognition policy that tries label-local barcode evidence before OCR and reports barcode hits separately from OCR recoveries.

Barcode-first recognition is a metric boundary as well as an implementation order: OCR can recover missed values, but it must not be counted as a barcode hit.

### Label-Local Evidence

Recognition evidence taken from the Stage1 label crop, a Stage2 field crop, or candidates derived inside that same label.

Label-local evidence excludes scanning the whole source photo because a single source photo can contain multiple physical labels.

### Quiet Zone

The blank margin around a barcode that lets a decoder identify where the barcode starts and ends without neighboring marks interfering.
