# IRH Web Interface

Web-based interface for Intrinsic Resonance Holography (IRH) computations.

## Architecture

- **Backend**: FastAPI REST API (`/backend`)
- **Frontend**: React/Vue application (`/frontend`) - Coming soon

## Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints
- `GET /` - API root and info
- `GET /health` - Health check

### Fixed Point & RG Flow
- `GET /api/v1/fixed-point` - Get Cosmic Fixed Point (Eq. 1.14)
- `POST /api/v1/rg-flow` - Integrate RG flow trajectory

### Observables
- `GET /api/v1/observables/C_H` - Universal exponent (Eq. 1.16)
- `GET /api/v1/observables/alpha` - Fine-structure constant (Eq. 3.4-3.5)
- `GET /api/v1/observables/dark-energy` - Dark energy w₀ (§2.3)
- `GET /api/v1/observables/liv` - Lorentz violation ξ (§2.5)

### Standard Model
- `GET /api/v1/standard-model/gauge-group` - SU(3)×SU(2)×U(1) derivation
- `GET /api/v1/standard-model/neutrinos` - Neutrino predictions

### Falsification
- `GET /api/v1/falsification/summary` - All testable predictions

## Theoretical Foundation

All computations are based on the IRH v21.1 Manuscript:
- Part 1: Sections 1-4 (Foundation, Spacetime, Standard Model)
- Part 2: Sections 5-8 + Appendices (QM, Cosmology, Falsification)

## Development

Phase 4.1 implementation per docs/ROADMAP.md.

Next steps:
- [ ] React frontend with interactive visualizations
- [ ] WebSocket support for real-time computation updates
- [ ] Celery task queue for long computations
- [ ] Docker containerization (Phase 4.2)
