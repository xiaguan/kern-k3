# SM103a SASS reference

`sm103a.json` is the machine-readable Blackwell ISA database for the B300 /
GB300 part (`sm_103a`) from https://github.com/kacper-daftcode/blackwell-isa
(MIT), copied verbatim at upstream commit cc2f62c379c3 (2026-09-19). It is
reverse-engineered, not NVIDIA's.

Top-level keys: `instructions` (14 879 instruction forms: 128-bit encoding
templates, operand fields, and per-form scheduling metadata such as pipeline
class, latency, throughput, stall defaults), `cost_model` (per-pipe issue
credits and IPC ceilings the scheduler uses), `operand_roles`, `stallfix`
(control-word stall rules), `_meta`.

Use it to read what `cuobjdump -sass build/<module>.cubin` or `nvdisasm`
prints: which pipe an instruction issues on, what it costs, and why a
sequence stalls. A browsable version is
https://kacper-daftcode.github.io/blackwell-isa/SM103A_ISA_REFERENCE.html.
