# Weight Witness v0.1 design

## Purpose and alternatives

Inspect an explicitly selected local safetensors checkpoint, validate its declared shard ownership, and compare tensor payloads without importing a model, executing pickle, or installing an ML runtime. The product is a portable evidence report, not a new tensor format, model evaluator, merger, or security certification. Native safetensors is the independent format reference; established tools remain preferable for loading models or tracking entire experiments.

The first release supports local files and Hugging Face-style `weight_map` indexes. No remote API, GPU, cloud service, repository crawl, or automatic data upload. Only synthetic public examples enter this repository. Public reports omit absolute source paths and arbitrary metadata values by default; tensor names may still be sensitive.

## Contracts

Python >=3.10, standard library only at runtime; MIT for original code. Package and module: `weight_witness`. Optional independent test oracle: numpy and safetensors, not runtime requirements. No claim that the namespace is reserved on PyPI; install from the repository or built wheel.

### Binary and index reader — `src/weight_witness/checkpoint.py`

Exports `FormatError(ValueError)`; frozen dataclass `TensorInfo(name: str, file: Path, dtype: str, shape: tuple[int,...], start: int, end: int)` with start/end absolute file byte offsets and computed `nbytes` and `numel` properties; dataclass `Checkpoint(tensors: dict[str,TensorInfo], files: tuple[Path,...], fingerprints: dict[Path,tuple], headers: dict[str,str])`; functions `load_checkpoint(path, *, max_header_bytes=16777216, max_tensors=100000, max_shards=4096) -> Checkpoint`, `iter_tensor_bytes(tensor, chunk_bytes=1048576) -> Iterator[bytes]`, `assert_unchanged(checkpoint) -> None`.

`headers` maps source basenames to SHA256 of exact header bytes. `fingerprints` records size, mtime_ns, inode and device, used to detect ordinary concurrent modification before/after scans; this is NOT an atomic snapshot against a hostile writer. Paths are internal only. Reader accepts an explicit `.safetensors` file, explicit `.safetensors.index.json`, or a directory containing `model.safetensors.index.json`, then `model.safetensors`, then exactly one `.safetensors` file. Ambiguous inputs are rejected. File and index symlinks and symlink shard components are rejected. Shards must be simple relative basenames (no nesting, slash, backslash, drive, colon, dot-dot or symlink); deliberately narrow scope.

Reject invalid UTF-8, duplicate JSON object keys, non-standard JSON constants, bool/negative offsets or dimensions, bad byte counts, out-of-bounds/overlap/gaps/trailing payload, unsupported dtype sizes, oversized header/index/tensor/shard budgets. Zero-element tensors and scalar shape `[]` are supported. Metadata must match safetensors string-map rules. Index weight_map is nonempty and exact: every declared tensor occurs in its declared file, every tensor in a referenced shard is declared once, and optional metadata.total_size equals unique payload byte total. Do not search unrelated shards; disclose referenced-shards-only coverage. No extraction, imports or writes.

### Compare engine — `src/weight_witness/compare.py`

Exports `compare_checkpoints(left, right, *, atol=0.0, rtol=0.0, chunk_bytes=1048576) -> dict` and `inspect_checkpoint(path, *, chunk_bytes=1048576) -> dict`. Left/right are paths; reader module is the only structural authority. `atol`/`rtol` finite nonnegative, bool rejected; chunk size positive integer and at most 64MiB. Deterministic ordering and no timestamp in report. Schema version `1`.

Diff report keys: `schema_version`, `kind='diff'`, `status` (`equal`, `close`, `different`), `policy` (atol,rtol,float_dtypes,other_dtypes), `summary` (total, equal, close, changed, added, removed, incompatible, nonfinite), `sources` (left/right with files basenames and header SHA256 maps), `tensors` list, `limitations` list. Each row: name, status (`equal`, `close`, `changed`, `added`, `removed`, `incompatible`), left/right descriptors or null (`dtype`,`shape`,`nbytes`,`sha256`), and `metrics` or null. Metrics for same-shape same-dtype F32/F16/BF16: count, finite_pairs, changed_values, outside_tolerance, nonfinite_left, nonfinite_right, max_abs, mean_abs, rmse, worst_index (flat C-order, null for empty/no finite pairs). Numerical policy: abs(right-left) <= atol + rtol*abs(left). Nonfinite values never pass, even when raw bytes match. +0/-0 can be `close` with zero numerical delta and different bytes. Other supported dtypes use exact byte equality only, with no numerical claim. SHA256 streaming covers every tensor payload, not arbitrary metadata. Schema/shape/dtype differences are structural differences, not rounded comparisons. overall equal iff every row equal; close iff only equal/close; different otherwise. Missing/added/incompatible count as differences. Empty checkpoints are allowed single files and called out by tensor count; empty indexes rejected.

Inspect report: schema_version, kind='inspect', status='valid' or 'nonfinite', sources, summary (tensors,files,bytes,nonfinite), tensors (name,dtype,shape,nbytes,sha256,nonfinite count or null), limitations. Only supported F32/F16/BF16 values scanned numerically; others byte-hashed only. No NaN/Infinity JSON output. Statistics calculated over finite pairs only, nonfinite counts separately exposed. If errors/budgets/unsupported formats prevent full scan, raise FormatError; CLI emits structured `status='error'` and exits 2, never a green partial report. Both source fingerprints checked around payload scans.

### Report — `src/weight_witness/report.py`

Exports `render_html(report: dict) -> str`. Standalone UTF-8 HTML, inline CSS/JS, no external network or fonts. Render inspect and diff; visible coverage and tolerance; searchable/filterable tensor evidence; expandable details; handles empty and all supported status states. Escape every untrusted string and script context, do not inject arbitrary JSON into executable script. Work without JS with all rows readable. No analytics, uploads or localStorage.

Visual direction: a graphite-on-frost instrument workbench, not a marketing dashboard. Palette frost #eef3f7, paper #ffffff, navy #17314d, ink #152333, teal #007d79, amber #9a5200, error #ad263c. Display uses system UI with condensed spacing; body Segoe UI/system; evidence Consolas/ui-monospace. Signature: a tensor-difference strip where each cell is one actual tensor and links to its evidence row. No fake charts or invented scalar performance. Keyboard focus, readable mobile layout and honest empty/error states.

### CLI, examples and release — main owner

`python -m weight_witness inspect PATH [--json FILE] [--html FILE]`; `diff LEFT RIGHT [--atol FLOAT --rtol FLOAT --chunk-bytes INT --json FILE --html FILE]`. stdout is JSON; 0 valid/equal/close, 1 differences or nonfinite, 2 input/operational errors. Outputs cannot overwrite inputs or shard files; no partial artifact written before a complete calculation. Synthetic demo generator creates left/right checkpoints with exact unchanged/changed/missing/dtype-change cases, plus expected bad index. README describes expected nonzero diff exit explicitly. Tests use unittest, CLI subprocesses, native safetensors oracle and actual browser interactions.

## Verification and stop conditions

Each unit starts with failing behavioral tests. Reviewer receives actual files, not just intended changes. Parent reruns complete suite, checks corrupt inputs and clean wheel install outside source tree, scans pending files for secrets/private paths, renders demo and exercises search/filter in browser. CI targets Linux/Windows and Python 3.10/3.12/3.13 with optional oracle job. Claims limited to observed checks; no real large-model benchmark, GPU validation, model-quality guarantee or user adoption claim. Publish only after blocker findings resolved. If authenticated publication is unavailable, leave release-ready artifacts and an exact blocker rather than fabricate an upload.
