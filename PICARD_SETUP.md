# Install and build the Picard starter

The archive is laid out relative to the repository root.

## Place the files

From the root of the `picard-iteration` checkout, copy:

- `src/ompl/picard/PicardIteration.h`
- `src/ompl/picard/README.md`
- `tests/picard/test_picard_iteration.cpp`
- `demos/PicardIteration.cpp`

Then apply the CMake patch:

```bash
git apply picard_cmake.patch
```

If you extracted this starter outside the repository, copy the four paths first
and run `git apply /path/to/picard_cmake.patch` from the repository root.

## Configure

```bash
cmake -S . -B build/picard -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOMPL_BUILD_VAMP=OFF \
  -DOMPL_BUILD_PYTHON_BINDINGS=OFF \
  -DOMPL_BUILD_TESTS=ON \
  -DOMPL_BUILD_DEMOS=ON
```

## Build only Picard targets

```bash
cmake --build build/picard -j "$(nproc)" --target \
  test_picard_iteration \
  demo_PicardIteration
```

## Run

```bash
ctest --test-dir build/picard --output-on-failure -R '^test_picard_iteration$'
./build/picard/demos/demo_PicardIteration
```

## Rebuild after header changes

```bash
cmake --build build/picard -j "$(nproc)" --target \
  test_picard_iteration demo_PicardIteration
```
