import { NextResponse } from "next/server";
import { readdir, stat } from "fs/promises";
import path from "path";

const ROOT = path.join(process.cwd(), "..");
const DATA_DIR = path.join(ROOT, "data");
const MIXED_DIR = path.join(ROOT, "data", "mixed");

const IMAGE_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);
const PERSON_NAMES = ["person1.JPG", "person1.jpg", "person1.jpeg", "person1.png"];

export async function GET() {
  try {
    let hasPersonImage = false;
    let mixedCount = 0;

    try {
      const dataFiles = await readdir(DATA_DIR);
      for (const name of dataFiles) {
        if (PERSON_NAMES.includes(name)) {
          const full = path.join(DATA_DIR, name);
          const st = await stat(full);
          if (st.isFile()) {
            hasPersonImage = true;
            break;
          }
        }
      }
    } catch {
      // data/ missing or not readable
    }

    try {
      const mixedFiles = await readdir(MIXED_DIR);
      for (const name of mixedFiles) {
        const ext = path.extname(name).toLowerCase();
        if (IMAGE_EXT.has(ext)) {
          const full = path.join(MIXED_DIR, name);
          const st = await stat(full);
          if (st.isFile()) mixedCount++;
        }
      }
    } catch {
      // data/mixed/ missing or not readable
    }

    return NextResponse.json({
      hasPersonImage,
      mixedCount,
      hasExistingImages: hasPersonImage || mixedCount > 0,
    });
  } catch (err) {
    console.error("Gallery status error:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Status failed" },
      { status: 500 }
    );
  }
}
