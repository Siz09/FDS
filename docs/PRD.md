# Product Requirements Document (PRD)
## Smart Event Gallery (SEG)

**Version:** 1.0  
**Last Updated:** February 2026  
**Status:** In Development

---

## 1. Executive Summary

### Problem Statement
People don't reject professional photos because of quality. They reject them because they can't get their photos easily or on time.

**Current Pain Points:**
- Photos arrive late (weeks after events)
- Buried inside thousands of images
- Guests lose interest before finding themselves
- Manual searching through folders is tedious

**Why Mobile Photos Win:**
- Instant delivery
- Personal and curated
- Effortless access

### Solution
Smart Event Gallery (SEG) is a guest-first photo access system for events that automatically finds and groups photos by person using face recognition.

**Core Promise:** "Get your event photos before the memory fades."

**Not real-time. Not perfect AI. Just fast, simple, and personal.**

---

## 2. Product Overview

### Target Users

**Primary Users:**
- **Event Guests** - People attending events who want their photos quickly
- **Event Photographers** - Professionals uploading event photos

**Secondary Users:**
- **Event Organizers** - Admins managing events and galleries
- **Event Hosts** - People creating events and inviting guests

### Use Cases

1. **Guest Photo Retrieval**
   - Guest scans QR code at event
   - Uploads selfie or reference photo
   - Receives personal gallery of all their photos
   - Downloads or shares instantly

2. **Multi-Person Photo Groups**
   - Guest adds multiple faces to their profile
   - Receives group photos containing any of those faces
   - Perfect for families or friend groups

3. **Photographer Upload**
   - Photographer uploads batch of event photos
   - System processes photos in background
   - Faces detected and indexed automatically

4. **Admin Management**
   - Create and manage events
   - Monitor processing status
   - View analytics and usage

---

## 3. Features

### 3.1 Core Features (MVP)

#### Guest Features
- **QR Code Scanning**
  - Scan QR code to access event gallery
  - Works in browser (no app download required)
  - Generates unique guest session

- **Selfie/Reference Photo Upload**
  - Upload single reference photo via web interface
  - Camera capture support (mobile)
  - File upload support (desktop)
  - Image validation and face detection feedback

- **Automatic Photo Matching**
  - Face detection and matching using MediaPipe + dlib
  - Configurable match tolerance (strict/normal/loose)
  - Multi-face detection per image
  - Match if ANY face in image matches reference

- **Personal Gallery**
  - View all matched photos in grid layout
  - See match confidence scores
  - Filter by date/time (if available)
  - Download individual photos
  - Download all photos as ZIP
  - Share gallery link

- **Multi-Face Support**
  - Add multiple reference faces to profile
  - Get photos containing any of those faces
  - Perfect for group photos

#### Photographer/Admin Features
- **Batch Photo Upload**
  - Upload multiple photos at once
  - Progress tracking
  - Background processing via job queue
  - Support for common formats (JPEG, PNG, WebP)

- **Event Management**
  - Create events with name, date, description
  - Generate unique QR codes per event
  - Set event visibility and access controls
  - Archive/delete events

- **Processing Status**
  - View processing queue status
  - See processing errors and retries
  - Monitor system health

#### System Features
- **Face Detection**
  - MediaPipe Face Detector (BlazeFace short-range)
  - Fast CPU-based detection
  - Multi-face detection per image
  - Configurable confidence thresholds

- **Face Embedding**
  - dlib face_recognition library
  - 128-dimensional embeddings
  - Euclidean distance matching
  - Configurable tolerance (default 0.6)

- **Vector Search**
  - PostgreSQL with pgvector extension (initial)
  - Optional migration to Qdrant for scale
  - Efficient similarity search
  - Indexed for performance

- **Background Processing**
  - Redis + BullMQ job queue
  - Async photo processing
  - Retry logic for failed jobs
  - Priority queue support

### 3.2 Future Features (Out of Scope for MVP)

- **Advanced Search**
  - Object detection (clothing, objects)
  - Scene recognition
  - Text search in metadata

- **Social Features**
  - Photo sharing between guests
  - Comments and reactions
  - Social media integration

- **Analytics**
  - Guest engagement metrics
  - Popular photos
  - Event attendance insights

- **Mobile Apps**
  - Native iOS app
  - Native Android app
  - Push notifications

- **AI Enhancements**
  - Automatic photo quality scoring
  - Automatic cropping and enhancement
  - Duplicate detection

- **Enterprise Features**
  - White-label branding
  - Custom domains
  - Advanced access controls
  - API access for integrations

- **Real-time Processing**
  - Live photo upload and matching
  - WebSocket updates
  - Real-time gallery updates

---

## 4. User Stories

### Guest Stories

**As a guest, I want to:**
1. Scan a QR code to access the event gallery so I don't need to remember URLs
2. Upload a selfie to find my photos so I don't have to search manually
3. See all my photos in one gallery so I can easily browse them
4. Download my photos so I can save them to my device
5. Share my gallery with others so they can see the photos too
6. Add multiple faces to get group photos so I can find photos with my friends/family

### Photographer Stories

**As a photographer, I want to:**
1. Upload photos in batches so I can process many photos efficiently
2. See processing status so I know when photos are ready
3. Manage events so I can organize photos by event
4. Generate QR codes so guests can easily access galleries

### Admin Stories

**As an admin, I want to:**
1. Create and manage events so I can organize photo collections
2. Monitor system health so I can ensure reliability
3. View analytics so I can understand usage patterns
4. Manage user access so I can control who sees what

---

## 5. Technical Requirements

### Performance
- Photo processing: < 5 seconds per photo (background)
- Face matching: < 2 seconds for 1000 photos
- Gallery load: < 1 second for initial render
- Image serving: < 500ms per image

### Scalability
- Support 10,000+ photos per event
- Handle 100+ concurrent users
- Process 1000+ photos/hour
- Store millions of face embeddings

### Reliability
- 99.9% uptime
- Automatic retry for failed processing
- Graceful degradation if face detection fails
- Data backup and recovery

### Security
- Guest sessions expire after 7 days (see Open Questions section for discussion)
- Rate limiting on upload endpoints
- Cloudflare Turnstile bot protection
- Secure image storage (Cloudinary for now / S3-compatible in future)
- No PII stored unnecessarily

### Privacy
- Guest sessions are anonymous
- Photos only accessible via QR code or direct link
- No tracking or analytics on guests
- GDPR-compliant data handling

---

## 6. Success Metrics

### User Engagement
- **Photo Retrieval Rate:** % of guests who successfully retrieve photos
- **Average Photos Per Guest:** Number of photos matched per guest
- **Time to First Match:** Time from upload to first matched photo
- **Gallery Views:** Number of times galleries are viewed

### System Performance
- **Processing Throughput:** Photos processed per hour
- **Match Accuracy:** % of correct matches (true positives)
- **False Positive Rate:** % of incorrect matches
- **System Uptime:** % of time system is available

### Business Metrics
- **Event Creation Rate:** Number of events created
- **Guest Retention:** % of guests who return for multiple events
- **Photographer Satisfaction:** NPS score from photographers

---

## 7. Out of Scope (Explicitly)

### Not Included in MVP
- ❌ Real-time photo processing (async only)
- ❌ Mobile native apps (web-only)
- ❌ Social media integration
- ❌ Photo editing/retouching
- ❌ Advanced analytics dashboard
- ❌ Multi-language support
- ❌ Payment processing
- ❌ User accounts/authentication for guests
- ❌ Email notifications
- ❌ SMS notifications
- ❌ Video support
- ❌ Live streaming
- ❌ Chat/messaging features
- ❌ Photo printing services
- ❌ Advanced AI features (object detection, scene recognition)

### Future Considerations
These may be added in future versions but are not part of the initial scope:
- Object detection for finding photos by clothing/objects
- Scene recognition for filtering by location/activity
- Advanced analytics and reporting
- Enterprise features (white-label, custom domains)
- API for third-party integrations

---

## 8. Constraints and Assumptions

### Constraints
- Must work in browser (no native app required)
- Must support common image formats (JPEG, PNG, WebP)
- Must work on mobile devices (responsive design)
- Must process photos asynchronously (not real-time)
- Must be cost-effective to run

### Assumptions
- Guests have smartphones with cameras
- Guests have internet connectivity
- Photos contain detectable faces
- Event organizers have basic technical knowledge
- Photographers can upload photos in batches

### Risks
- **Face Detection Failures:** Some photos may not have detectable faces
- **False Positives:** Incorrect matches may occur
- **Scale Challenges:** Large events may strain resources
- **Privacy Concerns:** Users may be concerned about face recognition

### Mitigation
- Provide manual search fallback
- Allow tolerance adjustment
- Implement horizontal scaling
- Clear privacy policy and opt-out options

---

## 9. Dependencies

### External Services
- **Object Storage:** Cloudinary (Current) / Cloudflare R2, MinIO, AWS S3, or Backblaze B2 (Future)
- **Database:** PostgreSQL with pgvector extension
- **Vector DB (Optional):** Qdrant for large-scale deployments
- **Job Queue:** Redis (Upstash or OSS)
- **CDN:** Cloudflare (for image delivery)
- **CAPTCHA:** Cloudflare Turnstile

### Third-Party Libraries
- **Face Detection:** MediaPipe Face Detector
- **Face Embedding:** dlib face_recognition
- **Image Processing:** OpenCV
- **QR Code:** html5-qrcode (browser)
- **Authentication:** Auth.js

---

## 10. Timeline and Milestones

### Phase 1: MVP (Current)
- ✅ Basic face detection and matching
- ✅ File-based photo storage
- ✅ Simple web interface
- ✅ QR code scanning
- 🔄 Database migration (PostgreSQL + pgvector)
- 🔄 Cloudinary storage integration
- 🔄 Job queue implementation
- 🔄 Admin dashboard

### Phase 2: Scale
- Vector database migration (Qdrant)
- Performance optimization
- Advanced error handling
- Monitoring and observability

### Phase 3: Enhancements
- Multi-face support improvements
- Advanced search features
- Analytics dashboard
- API documentation

---

## 11. Open Questions

1. **Storage Limits:** What are the storage limits per event?
2. **Retention Policy:** How long should photos be stored?
3. **Guest Sessions:** ~~How long should guest sessions last?~~ (Resolved: 7 days as specified in Security requirements)
4. **Match Tolerance:** What are the optimal tolerance values?
5. **Rate Limits:** What are appropriate rate limits for uploads?

---

## 12. Appendix

### Glossary
- **Face Embedding:** 128-dimensional vector representing a face
- **Tolerance:** Maximum distance for a match (lower = stricter)
- **Euclidean Distance:** L2 distance between embeddings
- **Match:** Photo containing a face within tolerance of reference
- **Gallery:** Collection of matched photos for a guest

### References
- MediaPipe Face Detector: https://github.com/google-ai-edge/mediapipe
- dlib face_recognition: https://github.com/ageitgey/face_recognition
- pgvector: https://github.com/pgvector/pgvector
- Qdrant: https://qdrant.tech/

---

**Document Owner:** Product Team  
**Reviewers:** Engineering, Design, Product  
**Approval Date:** TBD
