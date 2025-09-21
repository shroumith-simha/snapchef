# pathway_pipeline.py
"""
Safe shim for Pathway pipeline.

We avoid calling Pathway APIs at module import time because different
versions of the `pathway` package expose different APIs.
This file exposes a start_pipeline() function that will try to import
and run a Pathway pipeline if the required API is present. Otherwise
it simply returns without error so the Flask app can continue.
"""

import traceback

def start_pipeline():
    try:
        import pathway as pw
    except Exception:
        # Pathway not installed or import failed; no-op
        print("pathway not importable — skipping pipeline startup")
        return

    # Now check for a safe API surface. Some Pathway versions expose
    # pipeline helpers differently; we will attempt a minimal run.
    try:
        # If pw.run exists, call it — otherwise just log and return.
        if hasattr(pw, "run"):
            print("Pathway available — starting pw.run()")
            pw.run()
        else:
            print("pathway installed but no pw.run() — skipping pipeline startup")
    except Exception as e:
        print("Exception while starting Pathway pipeline:", e)
        traceback.print_exc()
        # swallow exceptions to avoid crashing the web process
        return
