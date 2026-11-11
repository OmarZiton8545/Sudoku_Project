# Data Dictionary — Project P7 (Handwritten Sudoku Reader and Solver)

## 1. Real Dataset: MNIST Handwritten Digits
- **Source**: MNIST (`torchvision.datasets.MNIST`, original: http://yann.lecun.com/exdb/mnist/)
- **Raw counts**: `60,000` training images, `10,000` test images (`28x28` grayscale `uint8`, centered digits).
- **Preprocessing (`make_data.py --seed 42`)**:
  1. Scale pixel intensities from `[0, 255]` to `[0.0, 1.0]` (`float32`).
  2. Drop digit `0` first (`5,923` dropped from train, `980` dropped from test), leaving `54,077` non-zero train images and `9,020` non-zero test images.
  3. Map digits `1–9` to class indices `0–8` via `data/labels.py` (`digit_to_index` / `index_to_digit`).
  4. Split the `54,077` non-zero training images with `--seed 42` into:
     - **Train**: `50,000` images (`y_train` class counts for digits `1–9`: `[6219, 5496, 5685, 5431, 5004, 5467, 5760, 5426, 5512]`)
     - **Validation**: `4,077` images (`y_val` class counts for digits `1–9`: `[523, 462, 446, 411, 417, 451, 505, 425, 437]`)
  5. **Test**: `9,020` images untouched (`y_test` class counts for digits `1–9`: `[1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]`).

### Processed Arrays (`data/processed/mnist_split.npz`)
| Key | Shape | Dtype | Real/Generated | Description |
| --- | --- | --- | --- | --- |
| `x_train` | `(50000, 28, 28)` | `float32` | Real (MNIST) | Scaled digit images `[0, 1]` |
| `y_train` | `(50000,)` | `int64` | Real (MNIST) | Mapped class index `0..8` (`digit - 1`) |
| `x_val` | `(4077, 28, 28)` | `float32` | Real (MNIST) | Scaled digit images `[0, 1]` |
| `y_val` | `(4077,)` | `int64` | Real (MNIST) | Mapped class index `0..8` (`digit - 1`) |
| `x_test` | `(9020, 28, 28)` | `float32` | Real (MNIST) | Scaled test digit images `[0, 1]` |
| `y_test` | `(9020,)` | `int64` | Real (MNIST) | Mapped class index `0..8` (`digit - 1`) |

## 2. Generated Data (`data/sudoku_generator.py`)
- **`solution`**: `9x9` `int64` grid (`1..9`), generated via randomized backtracking with a seeded `np.random.Generator`.
- **`puzzle`**: `9x9` `int64` grid (`0` for blank, `1..9` for givens), created by removing cells in random order while maintaining a unique solution (`count_solutions(puzzle, limit=2) == 1`).
- **`mosaic`**: `(252, 252)` grayscale `float32` image (tiled from `(9, 9, 28, 28)` cells) constructed **after** the NN is trained, with a dedicated generator `--seed` per run:
  - Blank cells (`puzzle == 0`) contain all-zero `28x28` black tiles, detected by `is_blank_cell(patch, threshold=0.01)` using mean pixel intensity (`< 0.01`; all non-zero MNIST test digits have mean `>= 0.0281`).
  - Normal given cells sample real MNIST test images from `easy_correct` (images correctly classified by the trained NN with confidence `>= 0.90`).
  - **Hard-digit injection rule**: To inject `k` misreads (`0..4` per puzzle), choose `k` given cells uniformly at random (using the run's generator seed) and replace each with a real MNIST test image of the **same true digit** from `hard_misread` (images that the trained NN classifies wrongly, without replacement within a puzzle). If `hard_misread` for a digit is exhausted or empty, fall back to `low_confidence` (images of that digit with the highest second-best NN probability / lowest confidence).
