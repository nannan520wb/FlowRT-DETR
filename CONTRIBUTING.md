# Contributing

Thank you for helping improve FlowRT-DETR. Bug fixes, documentation updates,
reproducibility reports, and well-scoped research extensions are welcome.

## Development workflow

1. Open an issue before a substantial behavioral change.
2. Create a focused branch and keep unrelated changes out of the pull request.
3. Install `requirements-dev.txt` and run:

   ```bash
   python -m compileall -q src optical_flow tools
   pytest -q tools/bbfr/test_bbfr.py tools/queue_estimation/test_queue.py
   ```

4. Document new configuration fields and include a minimal reproduction for a
   bug fix.
5. Do not commit datasets, checkpoints, generated predictions, or credentials.

By submitting a contribution, you agree that it may be distributed under this
repository's Apache-2.0 license.
