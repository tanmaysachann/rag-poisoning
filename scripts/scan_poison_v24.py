"""Run the V24 learned, containment, and optional identity scanner."""

from scripts.scan_poison_v22 import main


if __name__ == "__main__":
    main(default_integrity_mode="containment")
