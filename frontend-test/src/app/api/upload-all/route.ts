import { NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import path from "path";

const ROOT = path.join(process.cwd(), "..");
const MIXED_DIR = path.join(ROOT, "data", "mixed");

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const files = formData.getAll("files") as File[];
    if (!files?.length) {
      return NextResponse.json(
        { error: "No files" },
        { status: 400 }
      );
    }
    await mkdir(MIXED_DIR, { recursive: true });
    const saved: string[] = [];
    for (const file of files) {
      if (!file.size) continue;
      const name = file.name || `image-${Date.now()}.jpg`;
      const safeName = name.replace(/[^a-zA-Z0-9._-]/g, "_");
      const outPath = path.join(MIXED_DIR, safeName);
      const buffer = Buffer.from(await file.arrayBuffer());
      await writeFile(outPath, buffer);
      saved.push(`data/mixed/${safeName}`);
    }
    return NextResponse.json({ ok: true, saved, count: saved.length });
  } catch (err) {
    console.error("Upload-all error:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Upload failed" },
      { status: 500 }
    );
  }
}
