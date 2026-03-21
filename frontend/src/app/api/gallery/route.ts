import { NextResponse } from "next/server";
import { readdir, unlink, stat } from "fs/promises";
import path from "path";

const ROOT = path.join(process.cwd(), "..");
const MIXED_DIR = path.join(ROOT, "data", "mixed");

const IMAGE_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

export async function DELETE() {
  try {
    let deletedMixed = 0;
    try {
      const mixedFiles = await readdir(MIXED_DIR);
      for (const name of mixedFiles) {
        const ext = path.extname(name).toLowerCase();
        if (IMAGE_EXT.has(ext)) {
          const full = path.join(MIXED_DIR, name);
          const st = await stat(full);
          if (st.isFile()) {
            await unlink(full);
            deletedMixed++;
          }
        }
      }
    } catch {
      // data/mixed/ missing or not readable
    }
    return NextResponse.json({ ok: true, deletedMixed });
  } catch (err) {
    console.error("Gallery clear error:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Clear failed" },
      { status: 500 }
    );
  }
}
