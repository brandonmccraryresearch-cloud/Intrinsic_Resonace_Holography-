import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard.jsx';
import FixedPoint from './components/FixedPoint.jsx';
import RGFlow from './components/RGFlow.jsx';
import Observables from './components/Observables.jsx';
import StandardModel from './components/StandardModel.jsx';
import Falsification from './components/Falsification.jsx';
import './styles/App.css';

/**
 * IRH Web Application
 * 
 * THEORETICAL FOUNDATION: IRH v21.1 Manuscript
 * ROADMAP REFERENCE: docs/ROADMAP.md §4.1 - Web Interface
 */
function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <div className="header-logo">
            <h1>IRH</h1>
            <span className="version">v21.1</span>
          </div>
          <nav className="main-nav">
            <Link to="/" className="nav-link">Dashboard</Link>
            <Link to="/fixed-point" className="nav-link">Fixed Point</Link>
            <Link to="/rg-flow" className="nav-link">RG Flow</Link>
            <Link to="/observables" className="nav-link">Observables</Link>
            <Link to="/standard-model" className="nav-link">Standard Model</Link>
            <Link to="/falsification" className="nav-link">Falsification</Link>
          </nav>
        </header>

        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/fixed-point" element={<FixedPoint />} />
            <Route path="/rg-flow" element={<RGFlow />} />
            <Route path="/observables" element={<Observables />} />
            <Route path="/standard-model" element={<StandardModel />} />
            <Route path="/falsification" element={<Falsification />} />
          </Routes>
        </main>

        <footer className="app-footer">
          <p>
            IRH Computational Framework | 
            <a href="https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-" target="_blank" rel="noopener noreferrer">
              GitHub
            </a> | 
            <a href="/api/docs" target="_blank" rel="noopener noreferrer">
              API Docs
            </a>
          </p>
          <p className="citation">
            McCrary, B. D. (2025). <em>Intrinsic Resonance Holography v21.1: Computational Framework</em>
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;
