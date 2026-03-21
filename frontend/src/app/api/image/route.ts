import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import path from "path";

const ROOT = path.join(process.cwd(), "..");

const ALLOWED_PREFIX = path.resolve(ROOT, "data");

export async function GET(request: NextRequest) {
  const pathParam = request.nextUrl.searchParams.get("path");
  if (!pathParam) {
    return NextResponse.json({ error: "Missing path" }, { status: 400 });
  }
  const normalized = pathParam.replace(/^\/+/, "").replace(/\\/g, path.sep);
  const fullPath = path.isAbsolute(normalized)
    ? path.resolve(normalized)
    : path.resolve(ROOT, normalized);
  const allowedNormalized = path.resolve(ALLOWED_PREFIX);
  const fullNormalized = path.resolve(fullPath);
  const rel = path.relative(allowedNormalized, fullNormalized);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }
  try {
    const buffer = await readFile(fullPath);
    const ext = path.extname(fullPath).toLowerCase();
    const types: Record<string, string> = {
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".png": "image/png",
      ".gif": "image/gif",
      ".webp": "image/webp",
      ".bmp": "image/bmp",
    };
    const contentType = types[ext] ?? "application/octet-stream";
    return new NextResponse(buffer, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "private, max-age=60",
      },
    });
  } catch {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
}
