# Smart Gallery Frontend

Single-page dashboard: upload the **person to search**, upload **all images** to search in, then see **matched images** on the right.

## Run

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Flow

1. **Person to search** — Upload one reference image of the person. It is saved to `data/person1.JPG` (project root `data/`).
2. **Upload all images** — Upload multiple images. They are saved to `data/mixed/`.
3. **Matched images** — Click **Find matches**. The backend runs `enroll.py` (if needed) and `find_person.py`, then returns images that contain the person. They appear in the right column.

## Requirements

- Python face pipeline in the repo root: `scripts/enroll.py`, `scripts/find_person.py`, and dependencies (see main README). `find-matches` runs from the Next.js server and calls these scripts.
- Run the dev server from the `frontend` folder so `process.cwd()` resolves correctly; API routes use `..` to reach `data/`, `known/`, and `out/`.

## API

- `POST /api/person` — Upload one file; saved as `data/person1.JPG`.
- `POST /api/upload-all` — Upload multiple files; saved under `data/mixed/`.
- `POST /api/find-matches` — Run enroll (if no `known/person1.npy`) and find_person; returns `{ matches: [{ path, best_distance, num_faces }] }`.
- `GET /api/image?path=data/mixed/IMG_7452.JPG` — Serve an image from `data/` (for matched thumbnails).
