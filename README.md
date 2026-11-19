# Thaumaturge

A text-mode suite for programming GPUs and building models. Everything runs in a terminal, over SSH if you like,
inside tmux so a dropped connection loses nothing.

| Tool | What it is | State |
|------|------------|-------|
| **BitScryer** | An IDE for CUDA and hand-written PTX. Write a kernel in the left pane; the PTX and SASS it became show beside it, and each save builds it, runs it on the card, checks it against a CPU answer and times it against the card's top speed. Eight lessons lead from adding two lists to tensor cores. | Works |
| **Artificer** | A model builder: snap blocks together into a model, edit each block's code, rearrange them. | Planned ([docs/artificer.md](docs/artificer.md)) |
| **Sourcerer** | Version control. | Planned |
| **BitWalker** | A debugger that steps through low-level code. | Planned |

## BitScryer

```
bitscryer                        the lessons (it opens the one you were on)
bitscryer ~/kernels/mine         a project of your own: a folder with a .cu or .ptx file (and check.py, if any)
bitscryer new ~/kernels/mine     start a project: a kernel and a check.py to change
bitscryer --selftest             every lesson's answer built and run on the card: does BitScryer work here?
```

Ctrl-S saves, builds and runs. F1 opens the lesson, F2 gives a hint (only when you ask), F3 draws the blocks and
warps, F4 chooses the card, F9 lists every key. Where F-keys don't come through, Esc then a letter does the same.
More in [docs/bitscryer.md](docs/bitscryer.md).

### Needs

- Linux, Python 3.10 or newer, numpy, and tmux (optional, but recommended over SSH).
- An NVIDIA card (compute capability 6.0 or newer; lesson 8 needs tensor cores) and its driver.
- The CUDA toolkit (nvcc, ptxas, nvdisasm). Without a card, BitScryer still builds and shows PTX and SASS.

### Install

```
git clone https://github.com/Kaylachuuu/Thaumaturge
sudo sh Thaumaturge/install.sh
bitscryer --selftest
```

`install.sh` makes a small Python of its own in /opt/thaumaturge/venv and puts the `bitscryer` command in
/usr/local/bin; a `git pull` in the clone updates it. Or, without sudo: `pip install --user ./Thaumaturge`.
Run BitScryer as yourself, never as root. Your lessons and progress live in `~/bitscryer/`.

### Cards it leaves alone

By default BitScryer uses any card no other program has open. On a machine whose cards do other work, list
the ones to leave alone in `/etc/thaumaturge/bitscryer.json`: cards never to use, and cards lent out at times (busy
while a lock file is held or a service is running). See [examples/bitscryer.json](examples/bitscryer.json) and
`thaumaturge/bitscryer/settings.py`. The same file can say where CUDA is and which CPU cores builds may use.

### Tests

```
python3 tests/test_bitscryer.py
```

No card needed. With the CUDA toolkit installed it also builds every lesson.

## License

Apache License 2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE).
