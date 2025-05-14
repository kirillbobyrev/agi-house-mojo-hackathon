# 🏆🥈 Second place project in [AGI House x Modular GPU Kernel Hackathon](https://app.agihouse.org/events/modular-hackathon-20250510)

This project contains a highly efficient implementation of the device-wide
prefix sum algorithm using Mojo.

- `prefix_sum.mojo` contains the implementation of the prefix sum algorithm.
- Mojo stdlib patch to fix `block.reduce` (=> `block.sum` and other functions)
  and `block.prefix_sum`: https://github.com/modular/modular/pull/455

Running the benchmark for the prefix sum:

```
$ mojo prefix_sum.mojo
```

This will run both the prefix sum test and benchmark for it. Unfortunately, due
to the problems with `bench.Bench`, running an "actual" benchmark is not
possible. That is because the atomic dynamic block allocation counter is being
incremented during the warm-up and it's not possible to turn off warm-ups and
multiple sequential runs of the benchmark.
