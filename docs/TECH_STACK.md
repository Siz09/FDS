# Tech Stack Documentation
## Smart Event Gallery (SEG)

**Version:** 1.0  
**Last Updated:** February 2026

---

## Table of Contents
1. [Frontend Stack](#frontend-stack)
2. [Backend Stack](#backend-stack)
3. [API Layer](#api-layer)
4. [Authentication & Security](#authentication--security)
5. [Storage](#storage)
6. [Database](#database)
7. [Vector Search](#vector-search)
8. [Job Queue](#job-queue)
9. [CV / Face Processing](#cv--face-processing)
10. [Observability](#observability)
11. [Testing & Code Quality](#testing--code-quality)
12. [Development Tools](#development-tools)
13. [Deployment](#deployment)

---

## Frontend Stack

### Core Framework
- **Next.js:** `16.1.6`
  - App Router (not Pages Router)
  - Server Components by default
  - API Routes for backend logic
  - File-based routing

- **React:** `19.2.4` (or `19.2.3` if not using RSC packages)
  - Latest stable version
  - Server Components support
  - Concurrent features enabled
  - **Note:** When using React Server Component packages (react-server-dom-webpack, react-server-dom-parcel, react-server-dom-turbopack), use React `19.2.4` to address known vulnerabilities

- **TypeScript:** `5.x`
  - Strict mode enabled
  - No implicit any
  - Strict null checks

### UI Framework
- **Tailwind CSS:** `4.x`
  - Latest version
  - JIT mode enabled
  - Custom configuration

- **shadcn/ui:** Latest
  - Component library built on Radix UI
  - Tailwind CSS styling
  - Copy-paste components (not npm package)

### State Management & Data Fetching
- **TanStack Query (React Query):** `5.x`
  - Latest stable version
  - Server state management
  - Caching and synchronization
  - Polling for job status

### QR Code Scanning
- **html5-qrcode:** `2.x`
  - Browser-based QR code scanning
  - No app download required
  - Camera API support

### Form Validation
- **Zod:** `3.x`
  - Schema validation
  - Type inference
  - Runtime validation

### UI Components (Radix UI via shadcn)
- **@radix-ui/react-label:** `^2.1.8`
- **@radix-ui/react-select:** `^2.2.6`
- **@radix-ui/react-slot:** `^1.2.4`
- **@radix-ui/react-tabs:** `^1.1.13`

### Utilities
- **class-variance-authority:** `^0.7.1`
- **clsx:** `^2.1.1`
- **tailwind-merge:** `^3.4.0`
- **lucide-react:** `^0.563.0` (Icons)

### Development Dependencies
- **@tailwindcss/postcss:** `^4`
- **@types/node:** `^20`
- **@types/react:** `^19`
- **@types/react-dom:** `^19`
- **eslint:** `^9`
- **eslint-config-next:** `16.1.6`

---

## Backend Stack

### Runtime
- **Node.js:** `18.x` or `20.x` (LTS)
  - Minimum: 18.0.0
  - Recommended: 20.x LTS

### API Framework
- **Next.js API Routes:** `16.1.6`
  - Built-in API routes
  - TypeScript support
  - Server-side rendering

### Python Runtime
- **Python:** `3.10+`
  - Minimum: 3.10
  - Recommended: 3.11 or 3.12

### Python Web Framework
- **FastAPI:** `0.115.x` (for `python/face-service`)
  - Async support
  - OpenAPI documentation
  - Type hints

---

## API Layer

### API Documentation
- **OpenAPI Specification:** `3.1.0`
  - REST API documentation
  - Auto-generated from FastAPI
  - Swagger UI available

### API Standards
- **RESTful APIs**
  - Standard HTTP methods (GET, POST, PUT, DELETE)
  - JSON request/response format
  - Status codes (200, 201, 400, 401, 403, 404, 500)

---

## Authentication & Security

### Authentication Library
- **Auth.js (NextAuth.js):** `5.x`
  - Next.js authentication
  - Multiple providers support
  - Session management

### Token Management
- **JSON Web Tokens (JWT):** Via Auth.js
  - Short-lived guest sessions
  - Secure token generation
  - Token validation

### Bot Protection
- **Cloudflare Turnstile:** Latest
  - CAPTCHA alternative
  - Privacy-focused
  - No user interaction required

### Rate Limiting
- **Redis-based Rate Limiting**
  - Upstash Redis (cloud) OR
  - OSS Redis (self-hosted)
  - Sliding window algorithm
  - Per-endpoint limits

### Security Headers
- **Next.js Security Headers**
  - Content Security Policy
  - X-Frame-Options
  - X-Content-Type-Options
  - Strict-Transport-Security

---

## Storage

### Object Storage (S3-Compatible)
**Primary Options (choose one):**
- **Cloudflare R2:** Latest
  - S3-compatible API
  - No egress fees
  - Global CDN

- **MinIO:** Latest (self-hosted)
  - S3-compatible
  - Self-hosted option
  - Open source

- **AWS S3:** Latest
  - Standard S3 API
  - Mature and reliable
  - Pay-per-use

- **Backblaze B2:** Latest
  - S3-compatible API
  - Cost-effective
  - Good performance

### Storage Client Library
- **@aws-sdk/client-s3:** `3.x`
  - S3-compatible client
  - Works with all providers
  - TypeScript support

---

## Database

### Primary Database
- **PostgreSQL:** `15.x` or `16.x`
  - Minimum: 15.0
  - Recommended: 16.x
  - ACID compliance
  - JSON support

### ORM
- **Drizzle ORM:** `0.33.x` or latest
  - TypeScript-first ORM
  - Type-safe queries
  - Migration support

### Database Client
- **drizzle-orm/node-postgres:** Latest
  - PostgreSQL driver
  - Connection pooling
  - Transaction support

### Migration Tool
- **drizzle-kit:** Latest
  - Schema migrations
  - Type-safe migrations
  - Migration generation

---

## Vector Search

### Primary Vector Database
- **pgvector:** `0.8.1` (PostgreSQL extension)
  - Vector similarity search
  - HNSW indexing
  - IVFFlat indexing
  - L2 distance, cosine distance, inner product

### Alternative Vector Database (Optional)
- **Qdrant:** `1.9.x` or latest
  - Dedicated vector database
  - Better performance at scale
  - Python client available

### Qdrant Client
- **qdrant-client:** Latest (Python)
  - Python client library
  - REST API client
  - Async support

---

## Job Queue

### Queue System
- **BullMQ:** `5.x` or latest
  - Redis-based job queue
  - Job priorities
  - Retry logic
  - Job scheduling

### Redis
**Options:**
- **Upstash Redis:** Latest (cloud)
  - Serverless Redis
  - Global replication
  - REST API

- **Redis OSS:** `7.x` (self-hosted)
  - Open source
  - Self-hosted option
  - Standard Redis

### Redis Client
- **ioredis:** `5.x` (Node.js)
  - Redis client for Node.js
  - Connection pooling
  - Cluster support

---

## CV / Face Processing

### Face Detection
- **MediaPipe:** `0.10.x` or latest
  - Face Detector (BlazeFace)
  - Short-range model
  - CPU-based (fast)
  - Commercial-safe license

### Face Embedding
- **face_recognition:** `1.3.x` or latest
  - dlib-based face recognition
  - 128-dimensional embeddings
  - Euclidean distance matching
  - Commercial license required (check)

### Image Processing
- **OpenCV (opencv-python):** `4.8.x` or latest
  - Image I/O
  - Image preprocessing
  - Resizing and cropping
  - Format conversion

### Scientific Computing
- **NumPy:** `1.24.x` or latest
  - Numerical operations
  - Array manipulation
  - Vector operations

### Testing
- **pytest:** `7.0.x` or latest
  - Python testing framework
  - Fixtures and parametrization
  - Coverage reporting

---

## Observability

### Metrics
- **Prometheus:** Latest (compatible)
  - Metrics collection
  - Time-series database
  - Query language (PromQL)

### Visualization
- **Grafana:** Latest
  - Metrics visualization
  - Dashboards
  - Alerting

### Tracing
- **OpenTelemetry:** Latest
  - Distributed tracing
  - Instrumentation
  - Export to various backends

### OpenTelemetry Libraries
- **@opentelemetry/api:** Latest (Node.js)
- **@opentelemetry/sdk-node:** Latest
- **@opentelemetry/instrumentation-http:** Latest
- **opentelemetry-instrumentation-fastapi:** Latest (Python)

### Logging
- **Winston:** Latest (Node.js)
  - Structured logging
  - Multiple transports
  - Log levels

- **Python logging:** Built-in
  - Standard library
  - Configurable handlers
  - JSON formatting

---

## Testing & Code Quality

### End-to-End Testing
- **Playwright:** `1.40.x` or latest
  - Browser automation
  - Cross-browser testing
  - Mobile testing
  - Screenshot comparison

### Linting
- **ESLint:** `9.x`
  - JavaScript/TypeScript linting
  - Next.js config
  - Custom rules

### Type Checking
- **TypeScript:** `5.x`
  - Strict mode
  - Type checking
  - No implicit any

### Code Formatting
- **Prettier:** Latest (optional)
  - Code formatting
  - Consistent style
  - Editor integration

### Python Linting
- **ruff:** Latest (optional)
  - Fast Python linter
  - Import sorting
  - Code formatting

---

## Development Tools

### Package Management
- **npm:** `10.x` or latest
  - Node.js package manager
  - Workspace support

- **pip:** Latest
  - Python package manager
  - Virtual environment support

### Build Tools
- **Turbo:** `2.8.0` or latest
  - Monorepo build system
  - Task orchestration
  - Caching

### Version Control
- **Git:** Latest
  - Version control
  - Branching strategy
  - Commit conventions

### Containerization
- **Docker:** Latest
  - Container runtime
  - Multi-stage builds
  - Docker Compose for local dev

### Environment Management
- **dotenv:** Latest
  - Environment variables
  - `.env` file support
  - Type-safe env vars

---

## Deployment

### Hosting Platforms

**Frontend/API:**
- **Vercel:** Recommended
  - Next.js optimized
  - Edge functions
  - Automatic deployments

- **Railway:** Alternative
  - Full-stack hosting
  - Database included
  - Simple deployment

- **Fly.io:** Alternative
  - Global edge deployment
  - Docker-based
  - Good for Python services

**Python Services:**
- **Fly.io:** Recommended
  - Python support
  - Docker deployment
  - Global edge

- **Railway:** Alternative
  - Python support
  - Simple deployment
  - Database included

### CI/CD
- **GitHub Actions:** Latest
  - CI/CD pipelines
  - Automated testing
  - Deployment automation

### Monitoring
- **Sentry:** Latest (optional)
  - Error tracking
  - Performance monitoring
  - Release tracking

---

## Version Summary Table

| Component | Version | Purpose |
|-----------|---------|---------|
| Next.js | 16.1.6 | Frontend framework |
| React | 19.2.3 | UI library |
| TypeScript | 5.x | Type safety |
| Tailwind CSS | 4.x | Styling |
| TanStack Query | 5.x | Data fetching |
| Zod | 3.x | Validation |
| html5-qrcode | 2.x | QR scanning |
| Node.js | 18.x / 20.x | Runtime |
| Python | 3.10+ | Face processing |
| FastAPI | 0.115.x | Python API |
| PostgreSQL | 15.x / 16.x | Database |
| Drizzle ORM | 0.33.x | ORM |
| pgvector | 0.8.1 | Vector search |
| MediaPipe | 0.10.x | Face detection |
| face_recognition | 1.3.x | Face embedding |
| OpenCV | 4.8.x | Image processing |
| BullMQ | 5.x | Job queue |
| Redis | 7.x | Queue backend |
| Playwright | 1.40.x | E2E testing |
| ESLint | 9.x | Linting |
| Turbo | 2.8.0 | Build system |

---

## Installation Commands

### Frontend Dependencies
```bash
npm install next@16.1.6 react@19.2.3 react-dom@19.2.3
npm install @tanstack/react-query@latest zod@latest
npm install html5-qrcode@latest
npm install tailwindcss@4 @tailwindcss/postcss@4
npm install @radix-ui/react-label@^2.1.8 @radix-ui/react-select@^2.2.6
npm install class-variance-authority clsx tailwind-merge lucide-react
```

### Backend Dependencies
```bash
npm install drizzle-orm@latest drizzle-kit@latest
npm install @aws-sdk/client-s3@latest
npm install ioredis@latest bullmq@latest
npm install @opentelemetry/api@latest
```

### Python Dependencies
```bash
pip install mediapipe>=0.10.0
pip install face_recognition>=1.3.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install fastapi>=0.115.0
pip install pytest>=7.0.0
pip install qdrant-client  # Optional
```

### Database Setup
```sql
-- PostgreSQL with pgvector
CREATE EXTENSION vector;
```

---

## Compatibility Matrix

### Node.js Compatibility
- **Next.js 16:** Requires Node.js 18.17+ or 20.x
- **React 19:** Compatible with Node.js 18+

### Python Compatibility
- **MediaPipe:** Python 3.8+
- **face_recognition:** Python 3.7+
- **FastAPI:** Python 3.8+

### Browser Compatibility
- **Modern browsers:** Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Mobile:** iOS Safari 14+, Chrome Mobile 90+

---

## Upgrade Paths

### Next.js
- Current: 16.1.6
- Next major: 17.x (when released)
- Migration: Follow Next.js upgrade guide

### React
- Current: 19.2.3
- Next major: 20.x (when released)
- Migration: Follow React upgrade guide

### PostgreSQL
- Current: 15.x / 16.x
- Next major: 17.x (when released)
- Migration: Follow PostgreSQL upgrade guide

---

## Performance Targets

### Frontend
- **First Contentful Paint:** < 1.5s
- **Time to Interactive:** < 3s
- **Lighthouse Score:** > 90

### Backend
- **API Response Time:** < 200ms (p95)
- **Photo Upload:** < 2s per photo
- **Face Detection:** < 1s per photo
- **Face Matching:** < 2s for 1000 photos

### Database
- **Query Time:** < 50ms (p95)
- **Vector Search:** < 100ms (p95)

---

**Document Owner:** Engineering Team  
**Reviewers:** DevOps, Engineering  
**Last Reviewed:** TBD
