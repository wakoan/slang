# archive — gemma4_150 bring-up scaffolding

One-time scripts from the staged, bit-exact bring-up of the gemma4_150
("150 recipe") kernels. Kept for provenance; **not used by any runner**.

- `prep_stageN.py` — builds the input buffers + a numpy oracle for stage *N*.
- `validate_stageN.mjs` — runs the reference WGSL kernel in headless Chrome
  (CDP, localhost:8000) and diffs against the oracle.

The staged validation (stages 1–10) proved every reference kernel bit-exact on
real weights before the full runners were assembled. Once the runners existed
and were verified end-to-end, these became historical. See the main
`gemma4_150/README.md` for the process description.
