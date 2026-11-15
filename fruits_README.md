# `fruits.csv` — data for the Exotic Fruit Classification project

Data file read by
[`Exotic_Fruit_Classification_KNN.ipynb`](Exotic_Fruit_Classification_KNN.ipynb)
and by its Italian original in [`it/`](it/).

## Origin and licence

- **Provided by the course** (Master in AI Engineering) as the data set of the
  project; it was downloaded from the course's public link and is copied here
  byte for byte, without any change.
- **Licence: not declared.** The course does not state a licence or the
  origin of the data, and no other public source of this file could be found.
  The file is therefore **not covered by the MIT licence of this repository**;
  it is kept here only so that the notebook can be re-run reproducibly.
- Checksums: MD5 `1f5d3d3ec83265951fb8d4eee0f6d1c6`,
  SHA-256 `e6e8da803d5352d2c9871e0ba8fa67f4827eef4fb8c59c32b0209e58a69fed1b`
  (16 433 bytes).

## Structure

Comma-separated, ASCII, LF line endings, one header row and **500 rows**, no
missing values and no duplicated rows. Column names and class labels are in
Italian.

| Column | Meaning | Type |
|---|---|---|
| `Frutto` | fruit type (target): `Mela` apple, `Banana`, `Arancia` orange, `Uva` grape, `Kiwi` | text, 5 classes × 100 rows |
| `Peso (g)` | weight, grams | float |
| `Diametro medio (mm)` | average diameter, millimetres (as declared) | float |
| `Lunghezza media (mm)` | average length, millimetres (as declared) | float |
| `Durezza buccia (1-10)` | peel hardness on a declared 1–10 scale | float |
| `Dolcezza (1-10)` | sweetness on a declared 1–10 scale | float |

## Known problems

The values do not look like real measurements; the data set is most likely
**synthetic**. Checks that anyone can repeat with pandas:

- **Hardness outside its declared scale**: 40 of the 500 values of
  `Durezza buccia (1-10)` exceed 10 (maximum 13.72): 24 grapes, 15 bananas,
  1 apple.
- **Bananas: diameter and length look swapped.** For bananas the "diameter"
  ranges from about 120 to 300 mm and the "length" from about 1.6 to 4.3 mm —
  the reverse of a real banana. As a consequence a single feature (the
  diameter) separates bananas from every other class.
- **Lengths implausibly small for every fruit** (1.2–11.1 mm overall: e.g.
  apples 3–9 mm), as if they were centimetres rather than millimetres.
- **Weights implausibly low** for whole fruits (e.g. apples 31–111 g,
  bananas 13–47 g).
- **Perfectly balanced classes** (100 per class), unlike a real stream of
  incoming fruit.

These issues do not prevent the exercise (classifying five groups from five
numbers) but they matter for interpreting the results: an accuracy measured on
these data says little about how a model would behave on real fruit.
