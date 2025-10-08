# Installation

The installation is managed by `uv`. All the agent script that interfaces with `PyPantograph` needs to be run under this environment.

First, install the necessary dependencies:

```bash
bash install.sh
```

Then activate the uv environment under this current folder:
```bash
source .venv/bin/activate
```

## Verified Platforms

- Ubuntu 24.04 LTS

**Note**: this script has been verified on Ubuntu 24.04 LTS. For other OS, particularly MacOS, it has not been verified. 
The lean environment might have issue with Apple M-series chips as well. Consult installation guides of `PyPantograph` and `uv` for more details.