# Data card

- **Source:** https://github.com/hasaneyldrm/exercises-dataset (`data/exercises.json`, `data/exercises.schema.json`)
- **Commit:** `7455efae41b330c265e7cd4b78dfa848e7ce5ebd` (committed 2026-07-16), downloaded 2026-11-02
- **Size:** 1,324 exercises
- **Fields we use:** `name`, `body_part`, `target`, `equipment`, `secondary_muscles`
- **License:** data and text are MIT. Images/GIFs are © Gym visual (https://gymvisual.com/) and are not used here.

`data/raw/` is not committed (see `.gitignore`). To re-download the exact same files:

```bash
SHA=7455efae41b330c265e7cd4b78dfa848e7ce5ebd
for f in exercises.json exercises.schema.json; do
  curl -fsSL -o "data/raw/$f" "https://raw.githubusercontent.com/hasaneyldrm/exercises-dataset/$SHA/data/$f"
done
```
