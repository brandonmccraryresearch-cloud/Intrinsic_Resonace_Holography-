import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../api.js';

/**
 * RG Flow Component
 * 
 * Interactive RG flow integration from initial conditions to fixed point
 * 
 * THEORETICAL REFERENCE: IRH v21.1 §1.2, Eq. 1.12-1.13
 */
function RGFlow() {
  const [params, setParams] = useState({
    lambda_init: 30.0,
    gamma_init: 60.0,
    mu_init: 100.0,
    t_range: [-20.0, 10.0],
    n_steps: 500,
  });

  const [trajectory, setTrajectory] = useState(null);

  const mutation = useMutation({
    mutationFn: (flowParams) => apiClient.computeRGFlow(flowParams),
    onSuccess: (data) => {
      setTrajectory(data);
    },
  });

  const handleCompute = () => {
    mutation.mutate(params);
  };

  const handleParamChange = (key, value) => {
    setParams((prev) => ({ ...prev, [key]: parseFloat(value) }));
  };

  return (
    <div className="rg-flow-page">
      <header className="page-header">
        <h1>RG Flow Explorer</h1>
        <p className="reference">IRH v21.1 §1.2, Eq. 1.12-1.13</p>
      </header>

      <section className="theory-overview">
        <h2>β-Functions (Eq. 1.13)</h2>
        <div className="beta-equations">
          <div className="beta-eq">β_λ = -2λ̃ + (9/8π²)λ̃²</div>
          <div className="beta-eq">β_γ = (3/4π²)λ̃γ̃</div>
          <div className="beta-eq">β_μ = 2μ̃ + (1/2π²)λ̃μ̃</div>
        </div>
      </section>

      <section className="input-section">
        <h2>Initial Conditions</h2>
        <div className="input-grid">
          <div className="input-group">
            <label htmlFor="lambda_init">λ̃₀ (Initial quartic)</label>
            <input
              id="lambda_init"
              type="number"
              step="1"
              value={params.lambda_init}
              onChange={(e) => handleParamChange('lambda_init', e.target.value)}
            />
            <span className="hint">Fixed point: 52.64</span>
          </div>
          
          <div className="input-group">
            <label htmlFor="gamma_init">γ̃₀ (Initial QNCD)</label>
            <input
              id="gamma_init"
              type="number"
              step="1"
              value={params.gamma_init}
              onChange={(e) => handleParamChange('gamma_init', e.target.value)}
            />
            <span className="hint">Fixed point: 105.28</span>
          </div>
          
          <div className="input-group">
            <label htmlFor="mu_init">μ̃₀ (Initial holographic)</label>
            <input
              id="mu_init"
              type="number"
              step="1"
              value={params.mu_init}
              onChange={(e) => handleParamChange('mu_init', e.target.value)}
            />
            <span className="hint">Fixed point: 157.91</span>
          </div>
        </div>

        <div className="input-grid">
          <div className="input-group">
            <label htmlFor="n_steps">Integration steps</label>
            <input
              id="n_steps"
              type="number"
              step="100"
              value={params.n_steps}
              onChange={(e) => handleParamChange('n_steps', e.target.value)}
            />
          </div>
        </div>

        <button 
          onClick={handleCompute}
          disabled={mutation.isPending}
          className="compute-button"
        >
          {mutation.isPending ? 'Computing...' : 'Compute RG Flow'}
        </button>
      </section>

      {mutation.isError && (
        <section className="error-section">
          <p>Error: {mutation.error.message}</p>
        </section>
      )}

      {trajectory && (
        <section className="results-section">
          <h2>Flow Results</h2>
          
          <div className="convergence-status">
            <span className={trajectory.converged ? 'converged' : 'not-converged'}>
              {trajectory.converged ? '✓ Converged to Fixed Point' : '✗ Did Not Converge'}
            </span>
          </div>

          <div className="final-values">
            <h3>Final Values</h3>
            <div className="values-grid">
              <div className="value-item">
                <span className="label">λ̃_final</span>
                <span className="value">{trajectory.final_point[0]?.toFixed(6)}</span>
              </div>
              <div className="value-item">
                <span className="label">γ̃_final</span>
                <span className="value">{trajectory.final_point[1]?.toFixed(6)}</span>
              </div>
              <div className="value-item">
                <span className="label">μ̃_final</span>
                <span className="value">{trajectory.final_point[2]?.toFixed(6)}</span>
              </div>
            </div>
          </div>

          <div className="trajectory-data">
            <h3>Trajectory Data</h3>
            <p>{trajectory.trajectory?.length} points computed</p>
            <p>t range: [{trajectory.times?.[0]?.toFixed(2)}, {trajectory.times?.[trajectory.times.length-1]?.toFixed(2)}]</p>
            
            <div className="trajectory-preview">
              <h4>Sample Points (every 50th)</h4>
              <table>
                <thead>
                  <tr>
                    <th>t</th>
                    <th>λ̃</th>
                    <th>γ̃</th>
                    <th>μ̃</th>
                  </tr>
                </thead>
                <tbody>
                  {trajectory.trajectory
                    ?.filter((_, i) => i % 50 === 0 || i === trajectory.trajectory.length - 1)
                    .map((point, i) => (
                      <tr key={i}>
                        <td>{trajectory.times?.[i * 50 < trajectory.times.length ? i * 50 : trajectory.times.length - 1]?.toFixed(3)}</td>
                        <td>{point[0]?.toFixed(4)}</td>
                        <td>{point[1]?.toFixed(4)}</td>
                        <td>{point[2]?.toFixed(4)}</td>
                      </tr>
                    ))
                  }
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      <section className="notes-section">
        <h2>Computational Notes</h2>
        <div className="note-box">
          <h4>One-Loop vs Full Analysis</h4>
          <p>
            The RG flow computed here uses the one-loop β-functions (Eq. 1.13).
            Note that these one-loop equations have different zeros than the 
            Cosmic Fixed Point values (Eq. 1.14), which arise from the full 
            Wetterich equation analysis.
          </p>
          <ul>
            <li>β_λ = 0 at λ̃ = 16π²/9 ≈ 17.55 (one-loop)</li>
            <li>Cosmic Fixed Point: λ̃* = 48π²/9 ≈ 52.64 (full analysis)</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

export default RGFlow;
