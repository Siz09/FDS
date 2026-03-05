"use client";

import * as React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type MatchedImage = { path: string; best_distance: number; num_faces: number };
type GalleryStatus = {
  hasPersonImage: boolean;
  mixedCount: number;
  hasExistingImages: boolean;
};

export function Dashboard() {
  const [personFile, setPersonFile] = React.useState<File | null>(null);
  const [personPreview, setPersonPreview] = React.useState<string | null>(null);
  const [personSaved, setPersonSaved] = React.useState<string | null>(null);
  const [personLoading, setPersonLoading] = React.useState(false);
  const [personError, setPersonError] = React.useState<string | null>(null);

  const [uploadAllLoading, setUploadAllLoading] = React.useState(false);
  const [uploadAllSaved, setUploadAllSaved] = React.useState(0);
  const [uploadAllError, setUploadAllError] = React.useState<string | null>(null);

  const [matchedImages, setMatchedImages] = React.useState<MatchedImage[]>([]);
  const [findLoading, setFindLoading] = React.useState(false);
  const [findError, setFindError] = React.useState<string | null>(null);
  const [matchStrictness, setMatchStrictness] = React.useState<
    "strict" | "normal" | "loose"
  >("normal");

  const strictnessTolerance: Record<string, number> = {
    strict: 0.45,
    normal: 0.5,
    loose: 0.58,
  };

  const [galleryStatus, setGalleryStatus] =
    React.useState<GalleryStatus | null>(null);
  const [galleryStatusLoading, setGalleryStatusLoading] =
    React.useState(true);
  const [clearPersonLoading, setClearPersonLoading] = React.useState(false);
  const [clearMixedLoading, setClearMixedLoading] = React.useState(false);

  const personInputRef = React.useRef<HTMLInputElement>(null);
  const uploadAllInputRef = React.useRef<HTMLInputElement>(null);

  const fetchGalleryStatus = React.useCallback(async () => {
    try {
      const res = await fetch("/api/gallery-status");
      const data = await res.json();
      if (res.ok) setGalleryStatus(data);
    } catch {
      setGalleryStatus(null);
    } finally {
      setGalleryStatusLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchGalleryStatus();
  }, [fetchGalleryStatus]);

  const clearPersonImage = async () => {
    setClearPersonLoading(true);
    try {
      const res = await fetch("/api/person", { method: "DELETE" });
      if (!res.ok) throw new Error("Clear failed");
      setPersonSaved(null);
      setPersonFile(null);
      setPersonPreview(null);
      setMatchedImages([]);
      await fetchGalleryStatus();
    } finally {
      setClearPersonLoading(false);
    }
  };

  const clearUploadImages = async () => {
    setClearMixedLoading(true);
    try {
      const res = await fetch("/api/gallery", { method: "DELETE" });
      if (!res.ok) throw new Error("Clear failed");
      setUploadAllSaved(0);
      setMatchedImages([]);
      await fetchGalleryStatus();
    } finally {
      setClearMixedLoading(false);
    }
  };

  const handlePersonChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) return;
    setPersonFile(file);
    setPersonPreview(URL.createObjectURL(file));
    setPersonSaved(null);
    setPersonError(null);
    setMatchedImages([]);
    setPersonLoading(true);
    setPersonError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/person", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      setPersonSaved(data.path);
      await fetchGalleryStatus();
    } catch (err) {
      setPersonError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setPersonLoading(false);
    }
    e.target.value = "";
  };

  const handleUploadAllChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = e.target.files;
    if (!files?.length) return;
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (!list.length) return;
    setUploadAllError(null);
    setMatchedImages([]);
    setUploadAllLoading(true);
    try {
      const form = new FormData();
      list.forEach((f) => form.append("files", f));
      const res = await fetch("/api/upload-all", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      setUploadAllSaved(data.count ?? data.saved?.length ?? 0);
      await fetchGalleryStatus();
    } catch (err) {
      setUploadAllError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadAllLoading(false);
    }
    e.target.value = "";
  };

  const findMatches = async () => {
    setFindLoading(true);
    setFindError(null);
    setMatchedImages([]);
    try {
      const tolerance = strictnessTolerance[matchStrictness] ?? 0.5;
      const res = await fetch("/api/find-matches", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tolerance }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Find matches failed");
      setMatchedImages(data.matches ?? []);
    } catch (err) {
      setFindError(err instanceof Error ? err.message : "Find matches failed");
    } finally {
      setFindLoading(false);
    }
  };

  const imageUrl = (p: string) => `/api/image?path=${encodeURIComponent(p)}`;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border px-6 py-4">
        <h1 className="text-xl font-semibold tracking-tight">
          Smart Gallery — Person to Search
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Upload the person&apos;s image, then upload all images; matched images
          appear on the right.
        </p>
      </header>

      <main className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-6xl mx-auto">
          {/* Left: Person + Upload all */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Person to search</CardTitle>
                <CardDescription>
                  Upload one reference image of the person. Saves automatically
                  to <code className="text-xs bg-muted px-1 rounded">data/</code>
                  .
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {!galleryStatusLoading && galleryStatus?.hasPersonImage && (
                  <div className="rounded-md border border-amber-500/50 bg-amber-500/5 p-3 space-y-2">
                    <p className="text-sm text-muted-foreground">
                      There is already a person image in{" "}
                      <code className="text-xs bg-muted px-1 rounded">data/</code>
                      .
                    </p>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={clearPersonImage}
                      disabled={clearPersonLoading}
                    >
                      {clearPersonLoading ? "Clearing…" : "Clear person image"}
                    </Button>
                  </div>
                )}
                <div className="space-y-2">
                  <Label htmlFor="person-file">Choose image</Label>
                  <Input
                    id="person-file"
                    ref={personInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handlePersonChange}
                    disabled={personLoading}
                    className="cursor-pointer"
                  />
                </div>
                {personPreview && (
                  <div className="space-y-2">
                    <Label>Preview</Label>
                    <div className="relative w-40 h-40 rounded-lg border border-border overflow-hidden bg-muted">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={personPreview}
                        alt="Person"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    {personLoading && (
                      <p className="text-sm text-muted-foreground">
                        Saving…
                      </p>
                    )}
                  </div>
                )}
                {personSaved && !personLoading && (
                  <p className="text-sm text-muted-foreground rounded-md bg-muted/50 p-2">
                    Saved to <code className="text-xs">{personSaved}</code>
                  </p>
                )}
                {personError && (
                  <p className="text-sm text-destructive rounded-md bg-destructive/10 p-2">
                    {personError}
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Upload all images</CardTitle>
                <CardDescription>
                  Upload multiple images to search in. Saves automatically to{" "}
                  <code className="text-xs bg-muted px-1 rounded">
                    data/mixed/
                  </code>
                  .
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {!galleryStatusLoading && (galleryStatus?.mixedCount ?? 0) > 0 && (
                  <div className="rounded-md border border-amber-500/50 bg-amber-500/5 p-3 space-y-2">
                    <p className="text-sm text-muted-foreground">
                      There are already {galleryStatus?.mixedCount ?? 0} image
                      {(galleryStatus?.mixedCount ?? 0) !== 1 ? "s" : ""} in{" "}
                      <code className="text-xs bg-muted px-1 rounded">
                        data/mixed/
                      </code>
                      .
                    </p>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={clearUploadImages}
                      disabled={clearMixedLoading}
                    >
                      {clearMixedLoading
                        ? "Clearing…"
                        : "Clear upload images"}
                    </Button>
                  </div>
                )}
                <div className="space-y-2">
                  <Label htmlFor="upload-all-files">Choose images</Label>
                  <Input
                    id="upload-all-files"
                    ref={uploadAllInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={handleUploadAllChange}
                    disabled={uploadAllLoading}
                    className="cursor-pointer"
                  />
                </div>
                {uploadAllLoading && (
                  <p className="text-sm text-muted-foreground">Saving…</p>
                )}
                {uploadAllSaved > 0 && !uploadAllLoading && (
                  <p className="text-sm text-muted-foreground rounded-md bg-muted/50 p-2">
                    {uploadAllSaved} image(s) saved to data/mixed/
                  </p>
                )}
                {uploadAllError && (
                  <p className="text-sm text-destructive rounded-md bg-destructive/10 p-2">
                    {uploadAllError}
                  </p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right: Matched images */}
          <div>
            <Card className="sticky top-6">
              <CardHeader>
                <CardTitle>Matched images</CardTitle>
                <CardDescription>
                  Images that contain the person. Run find after saving person
                  and gallery images.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Match strictness</Label>
                  <Select
                    value={matchStrictness}
                    onValueChange={(value: "strict" | "normal" | "loose") =>
                      setMatchStrictness(value)
                    }
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Strictness" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="strict">
                        Strict (fewer false positives)
                      </SelectItem>
                      <SelectItem value="normal">Normal</SelectItem>
                      <SelectItem value="loose">Loose (more matches)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Stricter = only closer face matches count.
                  </p>
                </div>
                <Button
                  onClick={findMatches}
                  disabled={findLoading}
                  className="w-full"
                >
                  {findLoading ? "Finding…" : "Find matches"}
                </Button>
                {findError && (
                  <p className="text-sm text-destructive rounded-md bg-destructive/10 p-2">
                    {findError}
                  </p>
                )}
                {matchedImages.length > 0 && (
                  <div className="space-y-2">
                    <Badge variant="secondary">
                      {matchedImages.length} match
                      {matchedImages.length !== 1 ? "es" : ""}
                    </Badge>
                    <ul className="grid grid-cols-2 gap-2">
                      {matchedImages.map((m, i) => (
                        <li
                          key={`${m.path}-${i}`}
                          className="rounded-lg border border-border overflow-hidden bg-muted aspect-square"
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={imageUrl(m.path)}
                            alt={m.path}
                            className="w-full h-full object-cover"
                          />
                          <span className="block text-xs text-muted-foreground p-1 truncate">
                            d={m.best_distance.toFixed(2)}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {matchedImages.length === 0 && !findLoading && !findError && (
                  <p className="text-sm text-muted-foreground">
                    No matches yet. Save person and images, then click Find
                    matches.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
