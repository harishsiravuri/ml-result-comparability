"""Emit schemas/query_templates.json from the canonical registry (deterministic, $0).

The registry lives in src/certificates/query_templates.py; this script writes it to disk and
keeps the certificate schema's query.template enum in sync. Re-running is idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from certificates.query_templates import CERTIFICATE_TEMPLATE_ENUM, as_artifact  # noqa: E402

SCHEMAS = REPO / "schemas"


def main() -> None:
    (SCHEMAS / "query_templates.json").write_text(json.dumps(as_artifact(), indent=2) + "\n")
    print(f"wrote {SCHEMAS / 'query_templates.json'}")

    # verify (do not reformat) that the certificate schema enum is in sync. The schema is
    # hand-authored; the enum is edited surgically there and only checked here, so emitting the
    # registry never rewrites the schema's formatting.
    cert_path = SCHEMAS / "comparability_certificate.schema.json"
    cert = json.loads(cert_path.read_text())
    enum = cert["properties"]["query"]["properties"]["template"]["enum"]
    if set(enum) != set(CERTIFICATE_TEMPLATE_ENUM):
        raise SystemExit(
            f"certificate enum {enum} out of sync with registry {CERTIFICATE_TEMPLATE_ENUM}; "
            f"edit schemas/comparability_certificate.schema.json to match")
    print(f"certificate enum in sync: {enum}")


if __name__ == "__main__":
    main()
