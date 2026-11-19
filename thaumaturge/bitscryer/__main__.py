"""python -m thaumaturge.bitscryer: the same as the bitscryer command."""

import sys

from thaumaturge.bitscryer.screen import main

sys.exit(main(sys.argv[1:]))
