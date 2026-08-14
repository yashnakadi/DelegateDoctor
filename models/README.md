# Your models

This is your local workspace. Put your own model files here:

```
models/
├── model.py       your PyTorch model source
└── ...
```

Then run, from the repository root:

```bash
delegate-doctor optimize model.py
```

A bare filename is looked for in the current directory first, then here. An
explicit path (`delegate-doctor optimize projects/foo/model.py`) is always used
exactly as given.

Everything in this directory except this README is ignored by git, so your model
source, weights and exported programs are never committed.
