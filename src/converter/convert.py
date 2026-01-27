#!/usr/bin/env python3
"""
Convenience wrapper script for running the DICOM to NIfTI converter.
This allows running: python convert.py --input ... --output ...
"""

if __name__ == "__main__":
    from cli import main
    import sys
    sys.exit(main())
