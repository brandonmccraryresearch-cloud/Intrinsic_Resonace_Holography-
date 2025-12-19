import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api.js';

/**
 * Dashboard Component - Main landing page
 * 
 * Shows overview of IRH framework with key observables
 */
function Dashboard() {
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: apiClient.getHealth,
  });

  const fixedPointQuery = useQuery({
    queryKey: ['fixedPoint'],
    queryFn: apiClient.getFixedPoint,
  });

  const alphaQuery = useQuery({
    queryKey: ['alpha'],
    queryFn: apiClient.getFineStructureConstant,
  });

  const gaugeQuery = useQuery({
    queryKey: ['gaugeGroup'],
    queryFn: apiClient.getGaugeGroup,
  });

  return (
    <div className="dashboard">
      <section className="hero">
        <h1>Intrinsic Resonance Holography</h1>
        <p className="subtitle">
          A unified theory deriving all fundamental physical laws and constants 
          from quantum-informational first principles
        </p>
      </section>

      <section className="status-bar">
        {healthQuery.isLoading ? (
          <span className="status loading">Connecting to API...</span>
        ) : healthQuery.isError ? (
          <span className="status error">API Unavailable</span>
        ) : (
          <span className="status healthy">
            API Status: {healthQuery.data?.status} | 
            {healthQuery.data?.modules_loaded?.length || 0} modules loaded
          </span>
        )}
      </section>

      <div className="dashboard-grid">
        <div className="card key-prediction">
          <h3>Cosmic Fixed Point</h3>
          <p className="equation">Eq. 1.14</p>
          {fixedPointQuery.isLoading ? (
            <p>Loading...</p>
          ) : fixedPointQuery.isError ? (
            <p className="error">Error loading data</p>
          ) : (
            <div className="values">
              <div className="value-row">
                <span className="label">λ̃*</span>
                <span className="value">{fixedPointQuery.data?.lambda_star?.toFixed(6)}</span>
              </div>
              <div className="value-row">
                <span className="label">γ̃*</span>
                <span className="value">{fixedPointQuery.data?.gamma_star?.toFixed(6)}</span>
              </div>
              <div className="value-row">
                <span className="label">μ̃*</span>
                <span className="value">{fixedPointQuery.data?.mu_star?.toFixed(6)}</span>
              </div>
              <div className="value-row highlight">
                <span className="label">C_H</span>
                <span className="value">{fixedPointQuery.data?.C_H}</span>
              </div>
            </div>
          )}
        </div>

        <div className="card key-prediction">
          <h3>Fine-Structure Constant</h3>
          <p className="equation">Eqs. 3.4-3.5</p>
          {alphaQuery.isLoading ? (
            <p>Loading...</p>
          ) : alphaQuery.isError ? (
            <p className="error">Error loading data</p>
          ) : (
            <div className="values">
              <div className="value-row highlight">
                <span className="label">α⁻¹ (IRH)</span>
                <span className="value">{alphaQuery.data?.value}</span>
              </div>
              <div className="value-row">
                <span className="label">α⁻¹ (Exp)</span>
                <span className="value">137.035999084(21)</span>
              </div>
              <div className="value-row success">
                <span className="label">Agreement</span>
                <span className="value">
                  {alphaQuery.data?.details?.agreement ? '✓ Match' : '✗ Deviation'}
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="card key-prediction">
          <h3>Standard Model Emergence</h3>
          <p className="equation">§3.1, Appendix D</p>
          {gaugeQuery.isLoading ? (
            <p>Loading...</p>
          ) : gaugeQuery.isError ? (
            <p className="error">Error loading data</p>
          ) : (
            <div className="values">
              <div className="value-row highlight">
                <span className="label">Gauge Group</span>
                <span className="value">{gaugeQuery.data?.value}</span>
              </div>
              <div className="value-row">
                <span className="label">β₁</span>
                <span className="value">{gaugeQuery.data?.details?.betti_1}</span>
              </div>
              <div className="value-row">
                <span className="label">Generations</span>
                <span className="value">{gaugeQuery.data?.details?.fermion_generations}</span>
              </div>
              <div className="value-row">
                <span className="label">n_inst</span>
                <span className="value">{gaugeQuery.data?.details?.instanton_number}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <section className="theoretical-foundation">
        <h2>Theoretical Foundation</h2>
        <div className="foundation-grid">
          <div className="foundation-item">
            <h4>§1: cGFT Foundation</h4>
            <p>Quaternionic Group Field Theory on G_inf = SU(2) × U(1)_φ</p>
          </div>
          <div className="foundation-item">
            <h4>§2: Emergent Spacetime</h4>
            <p>d_spec: 2 → 4, Lorentzian signature, Einstein equations</p>
          </div>
          <div className="foundation-item">
            <h4>§3: Standard Model</h4>
            <p>Gauge groups, fermion masses, mixing matrices</p>
          </div>
          <div className="foundation-item">
            <h4>§4-8: Physics</h4>
            <p>QM emergence, cosmology, falsifiable predictions</p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Dashboard;
