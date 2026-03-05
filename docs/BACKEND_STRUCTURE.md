# Backend Structure Documentation
## Smart Event Gallery (SEG)

**Version:** 1.0  
**Last Updated:** February 2026

---

## Table of Contents
1. [Database Schema](#database-schema)
2. [API Endpoints](#api-endpoints)
3. [Data Models](#data-models)
4. [Job Queue Structure](#job-queue-structure)
5. [File Storage Structure](#file-storage-structure)
6. [Vector Search Schema](#vector-search-schema)
7. [Authentication & Authorization](#authentication--authorization)
8. [Error Handling](#error-handling)
9. [API Response Formats](#api-response-formats)

---

## Database Schema

### PostgreSQL Tables (Drizzle ORM)

#### Events Table
```typescript
import { pgTable, uuid, varchar, timestamp, boolean, pgEnum } from "drizzle-orm/pg-core";

export const eventStatusEnum = pgEnum("event_status", ["active", "archived", "deleted"]);

export const events = pgTable("events", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: varchar("name", { length: 255 }).notNull(),
  description: varchar("description", { length: 1000 }),
  eventDate: timestamp("event_date").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
  createdBy: uuid("created_by").notNull(), // References users.id
  status: eventStatusEnum("status").default("active").notNull(),
  isPublic: boolean("is_public").default(true).notNull(),
});
```

**Indexes:**
- `idx_events_created_by` on `created_by`
- `idx_events_status` on `status`
- `idx_events_event_date` on `event_date`

#### Users Table (Admin/Photographer)
```typescript
export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  name: varchar("name", { length: 255 }),
  passwordHash: varchar("password_hash", { length: 255 }), // For email/password auth
  role: varchar("role", { length: 50 }).default("photographer").notNull(), // photographer, admin
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
  lastLoginAt: timestamp("last_login_at"),
});
```

**Indexes:**
- `idx_users_email` on `email` (unique)
- `idx_users_role` on `role`

#### Guest Sessions Table
```typescript
export const guestSessions = pgTable("guest_sessions", {
  id: uuid("id").primaryKey().defaultRandom(),
  eventId: uuid("event_id").notNull().references(() => events.id),
  sessionToken: varchar("session_token", { length: 64 }).notNull().unique(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  expiresAt: timestamp("expires_at").notNull(), // 7 days from creation
  lastAccessedAt: timestamp("last_accessed_at").defaultNow().notNull(),
});
```

**Indexes:**
- `idx_guest_sessions_event_id` on `event_id`
- `idx_guest_sessions_token` on `session_token` (unique)
- `idx_guest_sessions_expires_at` on `expires_at`

#### Reference Faces Table
```typescript
export const referenceFaces = pgTable("reference_faces", {
  id: uuid("id").primaryKey().defaultRandom(),
  guestSessionId: uuid("guest_session_id").notNull().references(() => guestSessions.id),
  s3Key: varchar("s3_key", { length: 500 }).notNull(), // S3 key for reference image
  embedding: vector("embedding", { dimensions: 128 }).notNull(), // pgvector 128-d embedding
  createdAt: timestamp("created_at").defaultNow().notNull(),
});
```

**Indexes:**
- `idx_reference_faces_session_id` on `guest_session_id`
- `idx_reference_faces_embedding` using `hnsw` with `vector_l2_ops` // Vector index

#### Photos Table
```typescript
export const photos = pgTable("photos", {
  id: uuid("id").primaryKey().defaultRandom(),
  eventId: uuid("event_id").notNull().references(() => events.id),
  s3Key: varchar("s3_key", { length: 500 }).notNull(), // S3 key for photo
  s3ThumbnailKey: varchar("s3_thumbnail_key", { length: 500 }), // Thumbnail S3 key
  originalFilename: varchar("original_filename", { length: 255 }),
  mimeType: varchar("mime_type", { length: 50 }), // image/jpeg, image/png, etc.
  fileSize: integer("file_size"), // Bytes
  width: integer("width"),
  height: integer("height"),
  processingStatus: varchar("processing_status", { length: 50 }).default("pending").notNull(),
  // pending, processing, complete, error
  uploadedBy: uuid("uploaded_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  processedAt: timestamp("processed_at"),
});
```

**Indexes:**
- `idx_photos_event_id` on `event_id`
- `idx_photos_processing_status` on `processing_status`
- `idx_photos_created_at` on `created_at`

#### Face Embeddings Table
```typescript
export const faceEmbeddings = pgTable("face_embeddings", {
  id: uuid("id").primaryKey().defaultRandom(),
  photoId: uuid("photo_id").notNull().references(() => photos.id),
  embedding: vector("embedding", { dimensions: 128 }).notNull(), // pgvector 128-d embedding
  bboxX: integer("bbox_x").notNull(), // Bounding box coordinates
  bboxY: integer("bbox_y").notNull(),
  bboxWidth: integer("bbox_width").notNull(),
  bboxHeight: integer("bbox_height").notNull(),
  confidence: real("confidence").notNull(), // Detection confidence [0, 1]
  createdAt: timestamp("created_at").defaultNow().notNull(),
});
```

**Indexes:**
- `idx_face_embeddings_photo_id` on `photo_id`
- `idx_face_embeddings_embedding` using `hnsw` with `vector_l2_ops` // Vector index

#### Matched Photos Table (Junction Table)
```typescript
export const matchedPhotos = pgTable("matched_photos", {
  id: uuid("id").primaryKey().defaultRandom(),
  guestSessionId: uuid("guest_session_id").notNull().references(() => guestSessions.id),
  photoId: uuid("photo_id").notNull().references(() => photos.id),
  referenceFaceId: uuid("reference_face_id").references(() => referenceFaces.id),
  distance: real("distance").notNull(), // Euclidean distance
  matchedAt: timestamp("matched_at").defaultNow().notNull(),
});
```

**Indexes:**
- `idx_matched_photos_session_id` on `guest_session_id`
- `idx_matched_photos_photo_id` on `photo_id`
- `idx_matched_photos_distance` on `distance`
- Unique constraint on `(guest_session_id, photo_id)`

#### Job Queue Table (BullMQ uses Redis, but we track in DB)
```typescript
export const processingJobs = pgTable("processing_jobs", {
  id: uuid("id").primaryKey().defaultRandom(),
  jobId: varchar("job_id", { length: 255 }).notNull().unique(), // BullMQ job ID
  jobType: varchar("job_type", { length: 50 }).notNull(), // process-photo, match-photos
  status: varchar("status", { length: 50 }).default("pending").notNull(),
  // pending, processing, complete, failed, retrying
  payload: jsonb("payload"), // Job payload data
  result: jsonb("result"), // Job result data
  error: text("error"), // Error message if failed
  attempts: integer("attempts").default(0).notNull(),
  maxAttempts: integer("max_attempts").default(3).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
});
```

**Indexes:**
- `idx_processing_jobs_job_id` on `job_id` (unique)
- `idx_processing_jobs_status` on `status`
- `idx_processing_jobs_job_type` on `job_type`

---

## API Endpoints

### Guest Endpoints (Public, No Auth)

#### POST `/api/guest/{sessionToken}/reference-photo`
Upload reference photo for face matching.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```typescript
{
  success: true,
  referenceFaceId: string,
  s3Key: string,
  message: "Reference photo uploaded successfully"
}
```

**Errors:**
- `400`: No file provided
- `400`: Invalid file type
- `400`: No face detected
- `500`: Upload failed

---

#### GET `/api/guest/{sessionToken}/gallery`
Get matched photos for guest session.

**Request:**
- Method: `GET`
- Query params:
  - `limit?: number` (default: 100)
  - `offset?: number` (default: 0)
  - `sort?: "newest" | "oldest" | "best_match"` (default: "newest")

**Response:**
```typescript
{
  success: true,
  photos: [
    {
      id: string,
      s3Key: string,
      thumbnailUrl: string,
      fullUrl: string,
      distance: number,
      matchedAt: string,
      numFaces: number
    }
  ],
  total: number,
  limit: number,
  offset: number
}
```

**Errors:**
- `404`: Session not found
- `410`: Session expired

---

#### GET `/api/guest/{sessionToken}/gallery/status`
Get matching job status.

**Request:**
- Method: `GET`

**Response:**
```typescript
{
  success: true,
  status: "pending" | "processing" | "complete" | "error",
  progress?: number, // 0-100
  matchedCount?: number,
  totalPhotos?: number,
  error?: string
}
```

---

#### POST `/api/guest/{sessionToken}/reference-photos`
Add additional reference face (multi-face support).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```typescript
{
  success: true,
  referenceFaceId: string,
  message: "Reference face added. Re-matching photos..."
}
```

---

#### GET `/api/event/{eventId}/scan`
QR code scan entry point (creates guest session).

**Request:**
- Method: `GET`

**Response:**
- Redirects to `/event/{eventId}/guest/{sessionToken}`

---

### Landing / Waitlist (Public)

#### POST `/api/early-access`
Early access / waitlist signup (landing page form). Forwards to Google Apps Script; responses are appended to a Google Sheet.

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body:
```typescript
{
  fullName: string,
  email: string,
  role?: string,
  message?: string
}
```

**Response:**
```typescript
{ success: true }
```
- `400`: Missing name or email
- `502` / `503`: Google Script unavailable or misconfigured

**Implementation:** `apps/frontend-app/src/app/api/early-access/route.ts` → POST to `GOOGLE_SCRIPT_WEB_APP_URL` (env in frontend-app).

---

### Admin Endpoints (Auth Required)

#### POST `/api/admin/events`
Create new event.

**Request:**
- Method: `POST`
- Headers: `Authorization: Bearer {jwt_token}`
- Body:
```typescript
{
  name: string,
  description?: string,
  eventDate: string, // ISO 8601
  isPublic?: boolean
}
```

**Response:**
```typescript
{
  success: true,
  event: {
    id: string,
    name: string,
    eventDate: string,
    createdAt: string,
    qrCodeUrl: string
  }
}
```

**Errors:**
- `401`: Unauthorized
- `400`: Validation error

---

#### GET `/api/admin/events`
List all events (for authenticated user).

**Request:**
- Method: `GET`
- Headers: `Authorization: Bearer {jwt_token}`
- Query params:
  - `status?: "active" | "archived" | "deleted"`
  - `limit?: number`
  - `offset?: number`

**Response:**
```typescript
{
  success: true,
  events: [
    {
      id: string,
      name: string,
      eventDate: string,
      photoCount: number,
      guestSessionCount: number,
      status: string,
      createdAt: string
    }
  ],
  total: number
}
```

---

#### GET `/api/admin/events/{eventId}`
Get event details.

**Request:**
- Method: `GET`
- Headers: `Authorization: Bearer {jwt_token}`

**Response:**
```typescript
{
  success: true,
  event: {
    id: string,
    name: string,
    description: string,
    eventDate: string,
    photoCount: number,
    guestSessionCount: number,
    qrCodeUrl: string,
    createdAt: string
  }
}
```

---

#### POST `/api/admin/events/{eventId}/photos`
Upload photos to event.

**Request:**
- Method: `POST`
- Headers: `Authorization: Bearer {jwt_token}`
- Content-Type: `multipart/form-data`
- Body: `files` (array of image files)

**Response:**
```typescript
{
  success: true,
  uploaded: number,
  photos: [
    {
      id: string,
      s3Key: string,
      processingStatus: "pending"
    }
  ]
}
```

**Errors:**
- `401`: Unauthorized
- `404`: Event not found
- `400`: Invalid files

---

#### GET `/api/admin/events/{eventId}/photos`
List photos for event.

**Request:**
- Method: `GET`
- Headers: `Authorization: Bearer {jwt_token}`
- Query params:
  - `status?: "pending" | "processing" | "complete" | "error"`
  - `limit?: number`
  - `offset?: number`

**Response:**
```typescript
{
  success: true,
  photos: [
    {
      id: string,
      s3Key: string,
      thumbnailUrl: string,
      processingStatus: string,
      faceCount: number,
      createdAt: string
    }
  ],
  total: number
}
```

---

#### DELETE `/api/admin/events/{eventId}`
Delete event (soft delete).

**Request:**
- Method: `DELETE`
- Headers: `Authorization: Bearer {jwt_token}`

**Response:**
```typescript
{
  success: true,
  message: "Event deleted"
}
```

---

### Image Serving Endpoints

#### GET `/api/images/{photoId}`
Serve photo image (with authentication/authorization check).

**Request:**
- Method: `GET`
- Query params:
  - `size?: "thumbnail" | "medium" | "full"` (default: "full")
  - `format?: "webp" | "jpeg" | "png"` (default: "jpeg")

**Response:**
- Image file (binary)
- Headers:
  - `Content-Type: image/jpeg`
  - `Cache-Control: public, max-age=3600`

**Errors:**
- `404`: Photo not found
- `403`: Access denied

---

#### GET `/api/images/{photoId}/thumbnail`
Serve thumbnail (optimized endpoint).

**Request:**
- Method: `GET`

**Response:**
- Thumbnail image (binary)
- Headers:
  - `Content-Type: image/jpeg`
  - `Cache-Control: public, max-age=86400`

---

### Face Service Endpoints (Python FastAPI)

The Face Service is a standalone FastAPI app located in **`python/face-service/`**. It provides face detection, matching, and embedding. The frontend and Next.js API call it over HTTP (default port 8000). See `python/face-service/README.md` for setup and run instructions.

#### POST `/detect-face`
Detect faces in uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `image`: image file
  - `min_detection_confidence?: float` (default: 0.5)
  - `max_faces?: int` (default: 10)

**Response:**
```typescript
{
  num_faces: number,
  faces: [
    {
      x: number,
      y: number,
      w: number,
      h: number,
      confidence: number
    }
  ]
}
```

---

#### POST `/match-face`
Match faces between two images.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `reference`: reference image file
  - `target`: target image file
  - `tolerance?: float` (default: 0.6)

**Response:**
```typescript
{
  matched: boolean,
  best_distance: number,
  tolerance: number,
  num_faces_in_target: number
}
```

---

#### POST `/embed-face`
Generate face embedding from image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `image`: image file

**Response:**
```typescript
{
  embedding: number[], // 128-d array
  bbox: {
    x: number,
    y: number,
    w: number,
    h: number
  }
}
```

---

## Data Models

### TypeScript Interfaces

```typescript
// Event
interface Event {
  id: string;
  name: string;
  description?: string;
  eventDate: Date;
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  status: "active" | "archived" | "deleted";
  isPublic: boolean;
}

// Guest Session
interface GuestSession {
  id: string;
  eventId: string;
  sessionToken: string;
  createdAt: Date;
  expiresAt: Date;
  lastAccessedAt: Date;
}

// Reference Face
interface ReferenceFace {
  id: string;
  guestSessionId: string;
  s3Key: string;
  embedding: number[]; // 128-d vector
  createdAt: Date;
}

// Photo
interface Photo {
  id: string;
  eventId: string;
  s3Key: string;
  s3ThumbnailKey?: string;
  originalFilename?: string;
  mimeType?: string;
  fileSize?: number;
  width?: number;
  height?: number;
  processingStatus: "pending" | "processing" | "complete" | "error";
  uploadedBy?: string;
  createdAt: Date;
  processedAt?: Date;
}

// Face Embedding
interface FaceEmbedding {
  id: string;
  photoId: string;
  embedding: number[]; // 128-d vector
  bboxX: number;
  bboxY: number;
  bboxWidth: number;
  bboxHeight: number;
  confidence: number;
  createdAt: Date;
}

// Matched Photo
interface MatchedPhoto {
  id: string;
  guestSessionId: string;
  photoId: string;
  referenceFaceId?: string;
  distance: number;
  matchedAt: Date;
}
```

---

## Job Queue Structure

### BullMQ Job Types

#### `process-photo`
Process uploaded photo: detect faces and generate embeddings.

**Queue:** `photo-processing`
**Payload:**
```typescript
{
  photoId: string,
  eventId: string,
  s3Key: string
}
```

**Process:**
1. Download image from S3
2. Detect faces (MediaPipe)
3. Generate embeddings for each face (dlib)
4. Store embeddings in database
5. Generate thumbnail
6. Update photo status to "complete"

**Retry:** 3 attempts with exponential backoff

---

#### `match-photos`
Match photos against guest session reference faces.

**Queue:** `photo-matching`
**Payload:**
```typescript
{
  guestSessionId: string,
  eventId: string,
  tolerance: number
}
```

**Process:**
1. Get reference embeddings for session
2. Query vector DB for similar faces
3. Filter by tolerance
4. Create matched photos records
5. Update session last accessed time

**Retry:** 2 attempts

---

#### `generate-thumbnail`
Generate thumbnail for photo.

**Queue:** `thumbnail-generation`
**Payload:**
```typescript
{
  photoId: string,
  s3Key: string
}
```

**Process:**
1. Download image from S3
2. Resize to 300x300 (maintain aspect ratio)
3. Upload thumbnail to S3
4. Update photo record with thumbnail key

---

## File Storage Structure

### S3 Bucket Structure

```
bucket/
├── events/
│   └── {eventId}/
│       ├── photos/
│       │   ├── {photoId}.jpg          # Original photo
│       │   └── {photoId}_thumb.jpg    # Thumbnail
│       └── guests/
│           └── {sessionToken}/
│               └── reference.jpg      # Reference photo
```

### S3 Keys Format
- **Photo:** `events/{eventId}/photos/{photoId}.{ext}`
- **Thumbnail:** `events/{eventId}/photos/{photoId}_thumb.{ext}`
- **Reference:** `events/{eventId}/guests/{sessionToken}/reference.{ext}`

---

## Vector Search Schema

### pgvector Configuration

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector index on face embeddings
CREATE INDEX idx_face_embeddings_embedding 
ON face_embeddings 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Create vector index on reference faces
CREATE INDEX idx_reference_faces_embedding 
ON reference_faces 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);
```

### Vector Search Query

```sql
-- Find similar faces within tolerance
SELECT 
  fe.id,
  fe.photo_id,
  fe.embedding <-> $1::vector AS distance
FROM face_embeddings fe
WHERE fe.embedding <-> $1::vector < $2::float
ORDER BY fe.embedding <-> $1::vector
LIMIT 100;
```

---

## Authentication & Authorization

### Auth.js Configuration

**Session Strategy:** JWT
**Session Duration:** 
- Admin: 7 days
- Guest: 7 days (stored in database)

**Protected Routes:**
- `/admin/*` - Requires authentication
- `/api/admin/*` - Requires authentication

**Public Routes:**
- `/event/*` - Public access
- `/api/guest/*` - Public access (with session token)

### Authorization Levels

1. **Guest:** No authentication, session-based
2. **Photographer:** Can create events, upload photos
3. **Admin:** Full access to all features

---

## Error Handling

### Error Response Format

```typescript
{
  success: false,
  error: {
    code: string,        // Error code (e.g., "FACE_NOT_DETECTED")
    message: string,     // User-friendly message
    details?: any        // Additional error details
  }
}
```

### Error Codes

- `FACE_NOT_DETECTED`: No face found in image
- `SESSION_EXPIRED`: Guest session expired
- `SESSION_NOT_FOUND`: Invalid session token
- `EVENT_NOT_FOUND`: Event does not exist
- `PHOTO_NOT_FOUND`: Photo does not exist
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Insufficient permissions
- `VALIDATION_ERROR`: Request validation failed
- `PROCESSING_ERROR`: Photo processing failed
- `STORAGE_ERROR`: S3 storage error
- `DATABASE_ERROR`: Database operation failed

---

## API Response Formats

### Success Response
```typescript
{
  success: true,
  data: any,        // Response data
  meta?: {          // Optional metadata
    total?: number,
    limit?: number,
    offset?: number
  }
}
```

### Error Response
```typescript
{
  success: false,
  error: {
    code: string,
    message: string,
    details?: any
  }
}
```

### Pagination
```typescript
{
  success: true,
  data: any[],
  pagination: {
    total: number,
    limit: number,
    offset: number,
    hasMore: boolean
  }
}
```

---

**Document Owner:** Engineering Team  
**Reviewers:** Backend, DevOps, Engineering  
**Last Reviewed:** TBD
