import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api.js';

/**
 * Observables Component
 * 
 * Displays all computed physical constants and observables
 * 
 * THEORETICAL REFERENCE: IRH v21.1 §3.2, Eqs. 3.4-3.6
 */
function Observables() {
  const alphaQuery = useQuery({
    queryKey: ['alpha'],
    queryFn: apiClient.getFineStructureConstant,
  });

  const C_HQuery = useQuery({
    queryKey: ['C_H'],
    queryFn: apiClient.getUniversalExponent,
  });

  const darkEnergyQuery = useQuery({
    queryKey: ['darkEnergy'],
    queryFn: apiClient.getDarkEnergyEOS,
  });

  const livQuery = useQuery({
    queryKey: ['liv'],
    queryFn: apiClient.getLIVParameter,
  });

  const renderObservable = (query, formatValue = (v) => v) => {
    if (query.isLoading) return <span className="loading">Loading...</span>;
    if (query.isError) return <span className="error">Error</span>;
    return formatValue(query.data);
  };

  return (
    <div className="observables-page">
      <header className="page-header">
        <h1>Physical Observables</h1>
        <p className="reference">IRH v21.1 §3.2 (Constants), §2.3 (Dark Energy), §2.5 (LIV)</p>
      </header>

      <section className="observables-grid">
        {/* Fine-Structure Constant */}
        <div className="observable-card">
          <div className="observable-header">
            <h3>Fine-Structure Constant α⁻¹</h3>
            <span className="eq-ref">Eqs. 3.4-3.5</span>
          </div>
          {renderObservable(alphaQuery, (data) => (
            <div className="observable-content">
              <div className="main-value">
                <span className="label">IRH Prediction</span>
                <span className="value">{data.value}</span>
              </div>
              <div className="comparison">
                <div className="comp-row">
                  <span>Experimental</span>
                  <span>137.035999084(21)</span>
                </div>
                <div className="comp-row">
                  <span>Agreement</span>
                  <span className={data.details?.agreement ? 'success' : 'warning'}>
                    {data.details?.agreement ? '✓ Matches' : 'Deviation detected'}
                  </span>
                </div>
              </div>
              <p className="note">
                Derived from gauge coupling unification at the Cosmic Fixed Point
              </p>
            </div>
          ))}
        </div>

        {/* Universal Exponent C_H */}
        <div className="observable-card">
          <div className="observable-header">
            <h3>Universal Exponent C_H</h3>
            <span className="eq-ref">Eq. 1.16</span>
          </div>
          {renderObservable(C_HQuery, (data) => (
            <div className="observable-content">
              <div className="main-value highlight">
                <span className="label">C_H</span>
                <span className="value">{data.value}</span>
              </div>
              <div className="comparison">
                <div className="comp-row">
                  <span>Method</span>
                  <span>{data.details?.method || 'spectral'}</span>
                </div>
                <div className="comp-row">
                  <span>Ratio (3λ̃*/2γ̃*)</span>
                  <span>{data.details?.ratio_value}</span>
                </div>
              </div>
              <p className="note">
                First analytically computed constant of Nature - not fitted
              </p>
            </div>
          ))}
        </div>

        {/* Dark Energy w₀ */}
        <div className="observable-card">
          <div className="observable-header">
            <h3>Dark Energy w₀</h3>
            <span className="eq-ref">§2.3, Eqs. 2.21-2.23</span>
          </div>
          {renderObservable(darkEnergyQuery, (data) => (
            <div className="observable-content">
              <div className="main-value">
                <span className="label">w₀ (IRH)</span>
                <span className="value">{data.value?.toFixed(8)}</span>
              </div>
              <div className="uncertainty">
                <span>Uncertainty: ±{data.uncertainty?.toExponential(1)}</span>
              </div>
              <div className="comparison">
                <div className="comp-row">
                  <span>ΛCDM Value</span>
                  <span>-1.0</span>
                </div>
                <div className="comp-row">
                  <span>Deviation</span>
                  <span>{data.details?.deviation_from_lambda_cdm?.toFixed(4)}</span>
                </div>
                <div className="comp-row">
                  <span>Phantom?</span>
                  <span>{data.details?.is_phantom ? 'Yes' : 'No'}</span>
                </div>
              </div>
              <p className="note falsifiable">
                🔬 Falsifiable by Euclid/Roman if w₀ = -1.00 ± 0.01 confirmed
              </p>
            </div>
          ))}
        </div>

        {/* LIV Parameter */}
        <div className="observable-card">
          <div className="observable-header">
            <h3>Lorentz Invariance Violation ξ</h3>
            <span className="eq-ref">§2.5, Eqs. 2.24-2.26</span>
          </div>
          {renderObservable(livQuery, (data) => (
            <div className="observable-content">
              <div className="main-value">
                <span className="label">ξ (IRH)</span>
                <span className="value">{data.value?.toExponential(2)}</span>
              </div>
              <div className="comparison">
                <div className="comp-row">
                  <span>Formula</span>
                  <span>{data.details?.formula}</span>
                </div>
                <div className="comp-row">
                  <span>Current Bound</span>
                  <span>&lt; {data.details?.current_upper_bound}</span>
                </div>
                <div className="comp-row">
                  <span>CTA Sensitivity</span>
                  <span>{data.details?.cta_sensitivity}</span>
                </div>
              </div>
              <p className="note falsifiable">
                🔬 Falsifiable by CTA if ξ &lt; 10⁻⁵ established
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="derivation-note">
        <h2>Derivation Method</h2>
        <p>
          All observables are <strong>derived analytically</strong> from the Cosmic Fixed Point 
          couplings (Eq. 1.14), not fitted to experimental data. This represents a fundamental 
          departure from standard model parameterization.
        </p>
        <div className="derivation-flow">
          <span className="step">G_inf = SU(2)×U(1)_φ</span>
          <span className="arrow">→</span>
          <span className="step">cGFT Action</span>
          <span className="arrow">→</span>
          <span className="step">Cosmic Fixed Point</span>
          <span className="arrow">→</span>
          <span className="step">Physical Constants</span>
        </div>
      </section>
    </div>
  );
}

export default Observables;
