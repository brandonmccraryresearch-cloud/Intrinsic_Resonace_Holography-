# IRH Web Interface

Web-based interface for Intrinsic Resonance Holography (IRH) computations.

**Status**: Phase 4.1 - Frontend implementation complete ✅

## Architecture

```
webapp/
├── backend/           # FastAPI REST API (✅ Complete)
│   ├── app.py        # Main API application (13 endpoints)
│   ├── requirements.txt
│   └── tests/        # API tests
└── frontend/          # React + Vite frontend (✅ Complete)
    ├── src/
    │   ├── components/   # React components (6 pages)
    │   ├── styles/       # CSS styles
    │   ├── api.js       # API client
    │   └── App.jsx      # Main app with routing
    ├── package.json
    └── vite.config.js
```

## Quick Start

### Backend

```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Frontend

```bash
cd webapp/frontend
npm install
npm run dev
```

Frontend available at: http://localhost:3000

## Frontend Pages

| Page | Description | Reference |
|------|-------------|-----------|
| Dashboard | Overview with key observables | - |
| Fixed Point | Cosmic Fixed Point details | Eq. 1.14 |
| RG Flow | Interactive RG flow explorer | Eq. 1.12-1.13 |
| Observables | Physical constants (α⁻¹, C_H, w₀, ξ) | §3.2 |
| Standard Model | Gauge group and fermion emergence | §3.1 |
| Falsification | Testable predictions timeline | §7 |

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

**Completed**:
- [x] FastAPI backend with 13 endpoints
- [x] React frontend with 6 pages
- [x] Interactive RG flow explorer
- [x] Observables display
- [x] Falsification timeline

**Next steps (Phase 4.2)**:
- [ ] WebSocket support for real-time computation updates
- [ ] Celery task queue for long computations
- [ ] Docker containerization
- [ ] Cloud deployment (Kubernetes)
