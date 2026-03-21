import { NextResponse } from "next/server";
import { writeFile, mkdir, readdir, unlink, stat } from "fs/promises";
import path from "path";

const ROOT = path.join(process.cwd(), "..");
const DATA_DIR = path.join(ROOT, "data");
const KNOWN_REF = path.join(ROOT, "known", "person1.npy");
const PERSON_NAME = "person1";
const PERSON_FILE = `${PERSON_NAME}.JPG`;
const OUT_PATH = path.join(DATA_DIR, PERSON_FILE);
const PERSON_NAMES = ["person1.JPG", "person1.jpg", "person1.jpeg", "person1.png"];

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    if (!file || !file.size) {
      return NextResponse.json(
        { error: "No file or empty file" },
        { status: 400 }
      );
    }
    await mkdir(DATA_DIR, { recursive: true });
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    await writeFile(OUT_PATH, buffer);
    try {
      await unlink(KNOWN_REF);
    } catch {
      // known/person1.npy missing — next Find matches will create it
    }
    return NextResponse.json({
      ok: true,
      path: `data/${PERSON_FILE}`,
    });
  } catch (err) {
    console.error("Person upload error:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Upload failed" },
      { status: 500 }
    );
  }
}

export async function DELETE() {
  try {
    let deleted = 0;
    try {
      const dataFiles = await readdir(DATA_DIR);
      for (const name of dataFiles) {
        if (PERSON_NAMES.includes(name)) {
          const full = path.join(DATA_DIR, name);
          const st = await stat(full);
          if (st.isFile()) {
            await unlink(full);
            deleted++;
          }
        }
      }
    } catch {
      // data/ missing or not readable
    }
    try {
      await unlink(KNOWN_REF);
    } catch {
      // known/person1.npy missing
    }
    return NextResponse.json({ ok: true, deleted });
  } catch (err) {
    console.error("Person clear error:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Clear failed" },
      { status: 500 }
    );
  }
}
