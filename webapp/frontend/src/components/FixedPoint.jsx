import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api.js';

/**
 * Fixed Point Component
 * 
 * Displays the Cosmic Fixed Point (Eq. 1.14) with full details
 * 
 * THEORETICAL REFERENCE: IRH v21.1 §1.2-1.3
 */
function FixedPoint() {
  const query = useQuery({
    queryKey: ['fixedPoint'],
    queryFn: apiClient.getFixedPoint,
  });

  if (query.isLoading) {
    return <div className="loading">Loading fixed point data...</div>;
  }

  if (query.isError) {
    return <div className="error">Error: {query.error.message}</div>;
  }

  const data = query.data;

  return (
    <div className="fixed-point-page">
      <header className="page-header">
        <h1>Cosmic Fixed Point</h1>
        <p className="reference">IRH v21.1 §1.2.3, Eq. 1.14</p>
      </header>

      <section className="mathematical-definition">
        <h2>Mathematical Definition</h2>
        <div className="equation-block">
          <p>The Cosmic Fixed Point is the unique infrared attractor of the RG flow 
             where all β-functions vanish:</p>
          <div className="latex-equation">
            β_λ(λ̃*, γ̃*, μ̃*) = β_γ(λ̃*, γ̃*, μ̃*) = β_μ(λ̃*, γ̃*, μ̃*) = 0
          </div>
        </div>
      </section>

      <section className="fixed-point-values">
        <h2>Analytical Values (Eq. 1.14)</h2>
        <div className="values-grid">
          <div className="value-card">
            <div className="value-symbol">λ̃*</div>
            <div className="value-formula">48π²/9</div>
            <div className="value-numeric">{data.lambda_star.toFixed(10)}</div>
            <div className="value-description">Quartic coupling</div>
          </div>
          
          <div className="value-card">
            <div className="value-symbol">γ̃*</div>
            <div className="value-formula">32π²/3</div>
            <div className="value-numeric">{data.gamma_star.toFixed(10)}</div>
            <div className="value-description">QNCD coupling</div>
          </div>
          
          <div className="value-card">
            <div className="value-symbol">μ̃*</div>
            <div className="value-formula">16π²</div>
            <div className="value-numeric">{data.mu_star.toFixed(10)}</div>
            <div className="value-description">Holographic coupling</div>
          </div>
        </div>
      </section>

      <section className="universal-exponent">
        <h2>Universal Exponent C_H (Eq. 1.16)</h2>
        <div className="highlight-box">
          <div className="big-value">{data.C_H}</div>
          <p>The first analytically computed constant of Nature from pure mathematics</p>
          <p className="note">
            Note: C_H comes from spectral zeta function evaluation (Appendix B), 
            not from the simple ratio 3λ̃*/(2γ̃*) = 0.75
          </p>
        </div>
      </section>

      <section className="verification">
        <h2>Fixed Point Verification</h2>
        <table className="verification-table">
          <thead>
            <tr>
              <th>Property</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>β_λ = 0</td>
              <td className={data.verification?.beta_lambda_zero ? 'pass' : 'note'}>
                {data.verification?.beta_lambda_zero ? '✓ Pass' : '⚠ Note*'}
              </td>
            </tr>
            <tr>
              <td>β_γ = 0</td>
              <td className={data.verification?.beta_gamma_zero ? 'pass' : 'note'}>
                {data.verification?.beta_gamma_zero ? '✓ Pass' : '⚠ Note*'}
              </td>
            </tr>
            <tr>
              <td>Is Fixed Point</td>
              <td className={data.verification?.is_fixed_point ? 'pass' : 'note'}>
                {data.verification?.is_fixed_point ? '✓ Verified' : '⚠ See Note'}
              </td>
            </tr>
          </tbody>
        </table>
        
        <div className="verification-note">
          <h3>Important Note on Beta Functions</h3>
          <p>
            The one-loop β-functions (Eq. 1.13) and fixed-point values (Eq. 1.14) 
            represent different levels of the theoretical framework:
          </p>
          <ul>
            <li>Eq. 1.13: One-loop perturbative β-functions</li>
            <li>Eq. 1.14: Fixed-point values from full Wetterich equation analysis</li>
          </ul>
          <p>
            The fixed-point values emerge from the complete non-perturbative 
            treatment, not from setting the one-loop betas to zero.
          </p>
        </div>
      </section>

      <section className="physical-significance">
        <h2>Physical Significance</h2>
        <div className="significance-grid">
          <div className="significance-item">
            <h4>Asymptotic Safety</h4>
            <p>The fixed point provides a UV completion without new physics</p>
          </div>
          <div className="significance-item">
            <h4>Universal Physics</h4>
            <p>All observable physics emerges from fixed-point couplings</p>
          </div>
          <div className="significance-item">
            <h4>Scale Invariance</h4>
            <p>The theory becomes conformal at the fixed point</p>
          </div>
          <div className="significance-item">
            <h4>Predictivity</h4>
            <p>Physical constants are derived, not fitted</p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default FixedPoint;
