import { NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import { readFile } from "fs/promises";

const ROOT = path.join(process.cwd(), "..");
const KNOWN_DIR = path.join(ROOT, "known");
const REF_PATH = path.join(KNOWN_DIR, "person1.npy");
const MIXED_DIR = path.join(ROOT, "data", "mixed");
const REPORT_PATH = path.join(ROOT, "out", "report.json");

const DEFAULT_TOLERANCE = 0.5;
const MIN_DETECTION_CONFIDENCE = 0.6;
const NUM_JITTERS = 2;

const isWin = process.platform === "win32";
const VENV_PYTHON = isWin
    ? path.join(ROOT, ".venv", "Scripts", "python.exe")
    : path.join(ROOT, ".venv", "bin", "python");

function runEnroll(): Promise<void> {
    return new Promise((resolve, reject) => {
        const personImage = path.join(ROOT, "data", "person1.JPG");
        const child = spawn(
            VENV_PYTHON,
            [
                path.join(ROOT, "face-service", "scripts", "enroll.py"),
                "--name",
                "person1",
                "--image",
                personImage,
                "--out",
                REF_PATH,
                "--min-detection-confidence",
                String(MIN_DETECTION_CONFIDENCE),
                "--num-jitters",
                String(NUM_JITTERS),
            ],
            { cwd: ROOT, shell: false } // 'shell: false' is safer with direct executable path
        );
        let stderr = "";
        child.stderr?.on("data", (d) => { stderr += d.toString(); });
        child.on("close", (code) => {
            if (code === 0) resolve();
            else reject(new Error(stderr || `enroll exited ${code}`));
        });
        child.on("error", (err) => {
            // Fallback to simpler 'python' if venv fails, or just reject with clearer error
            if ((err as any).code === 'ENOENT') {
                reject(new Error(`Python executable not found at ${VENV_PYTHON}. Did you create the .venv?`));
            } else {
                reject(err);
            }
        });
    });
}

function runFindPerson(tolerance: number): Promise<void> {
    return new Promise((resolve, reject) => {
        const child = spawn(
            VENV_PYTHON,
            [
                path.join(ROOT, "face-service", "scripts", "find_person.py"),
                "--name",
                "person1",
                "--ref",
                REF_PATH,
                "--folder",
                MIXED_DIR,
                "--tolerance",
                String(tolerance),
                "--min-detection-confidence",
                String(MIN_DETECTION_CONFIDENCE),
                "--num-jitters",
                String(NUM_JITTERS),
                "--report",
                REPORT_PATH,
            ],
            { cwd: ROOT, shell: false }
        );
        let stderr = "";
        child.stderr?.on("data", (d) => { stderr += d.toString(); });
        child.on("close", (code) => {
            if (code === 0) resolve();
            else reject(new Error(stderr || `find_person exited ${code}`));
        });
        child.on("error", reject);
    });
}

export async function POST(request: Request) {
    let tolerance = DEFAULT_TOLERANCE;
    try {
        const body = await request.json().catch(() => ({}));
        if (
            typeof body.tolerance === "number" &&
            body.tolerance >= 0.35 &&
            body.tolerance <= 0.7
        ) {
            tolerance = body.tolerance;
        }
    } catch {
        // use default
    }

    try {
        const refExists = await import("fs/promises")
            .then((fs) => fs.access(REF_PATH).then(() => true).catch(() => false));
        if (!refExists) {
            await runEnroll();
        }
        await runFindPerson(tolerance);
        const raw = await readFile(REPORT_PATH, "utf-8");
        const report = JSON.parse(raw) as {
            person: string;
            matches: Array<{ path: string; best_distance: number; num_faces: number }>;
            all_results?: Array<{ path: string; matched: boolean }>;
        };
        const matches = report.matches || [];
        return NextResponse.json({
            ok: true,
            matches: matches.map((m) => {
                const full = m.path.replace(/\\/g, path.sep);
                const relative = path.relative(ROOT, path.resolve(ROOT, full));
                const pathForApi = relative.replace(/\\/g, "/");
                return {
                    path: pathForApi,
                    best_distance: m.best_distance,
                    num_faces: m.num_faces,
                };
            }),
        });
    } catch (err) {
        console.error("Find matches error:", err);
        return NextResponse.json(
            {
                error: err instanceof Error ? err.message : "Find matches failed",
            },
            { status: 500 }
        );
    }
}
