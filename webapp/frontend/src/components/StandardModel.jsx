import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api.js';

/**
 * Standard Model Component
 * 
 * Displays Standard Model emergence from topology
 * 
 * THEORETICAL REFERENCE: IRH v21.1 §3.1-3.2, Appendix D
 */
function StandardModel() {
  const gaugeQuery = useQuery({
    queryKey: ['gaugeGroup'],
    queryFn: apiClient.getGaugeGroup,
  });

  const neutrinoQuery = useQuery({
    queryKey: ['neutrinos'],
    queryFn: apiClient.getNeutrinoPredictions,
  });

  return (
    <div className="standard-model-page">
      <header className="page-header">
        <h1>Standard Model Emergence</h1>
        <p className="reference">IRH v21.1 §3.1-3.2, Appendix D</p>
      </header>

      <section className="emergence-overview">
        <h2>From Topology to Particle Physics</h2>
        <p>
          The Standard Model gauge structure and matter content emerge from 
          topological invariants of the manifold M³ = G_inf / Γ_R:
        </p>
        <div className="emergence-chain">
          <div className="chain-item">
            <span className="symbol">β₁ = 12</span>
            <span className="meaning">First Betti number</span>
            <span className="result">→ SU(3)×SU(2)×U(1)</span>
          </div>
          <div className="chain-item">
            <span className="symbol">n_inst = 3</span>
            <span className="meaning">Instanton number</span>
            <span className="result">→ 3 fermion generations</span>
          </div>
        </div>
      </section>

      <section className="gauge-group-section">
        <h2>Gauge Group Derivation</h2>
        {gaugeQuery.isLoading ? (
          <p>Loading...</p>
        ) : gaugeQuery.isError ? (
          <p className="error">Error loading gauge group data</p>
        ) : (
          <div className="gauge-details">
            <div className="main-result">
              <h3>{gaugeQuery.data?.value}</h3>
              <p className="eq-ref">{gaugeQuery.data?.theoretical_reference}</p>
            </div>
            
            <div className="decomposition">
              <h4>Generator Decomposition</h4>
              <table>
                <thead>
                  <tr>
                    <th>Group</th>
                    <th>Generators</th>
                    <th>Role</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>SU(3)</td>
                    <td>{gaugeQuery.data?.details?.su3_generators}</td>
                    <td>Strong force (QCD)</td>
                  </tr>
                  <tr>
                    <td>SU(2)</td>
                    <td>{gaugeQuery.data?.details?.su2_generators}</td>
                    <td>Weak isospin</td>
                  </tr>
                  <tr>
                    <td>U(1)</td>
                    <td>{gaugeQuery.data?.details?.u1_generators}</td>
                    <td>Hypercharge</td>
                  </tr>
                  <tr className="total-row">
                    <td><strong>Total</strong></td>
                    <td><strong>{gaugeQuery.data?.details?.total_generators}</strong></td>
                    <td><strong>= β₁</strong></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="fermion-generations">
        <h2>Fermion Generations</h2>
        {gaugeQuery.isLoading ? (
          <p>Loading...</p>
        ) : gaugeQuery.isError ? (
          <p className="error">Error loading data</p>
        ) : (
          <div className="generations-box">
            <div className="gen-value">
              <span className="number">{gaugeQuery.data?.details?.fermion_generations}</span>
              <span className="label">Generations</span>
            </div>
            <p>Derived from n_inst = {gaugeQuery.data?.details?.instanton_number}</p>
            <p className="note">
              This is not an input parameter - it emerges from the topology
            </p>
          </div>
        )}
      </section>

      <section className="neutrino-section">
        <h2>Neutrino Sector Predictions</h2>
        {neutrinoQuery.isLoading ? (
          <p>Loading...</p>
        ) : neutrinoQuery.isError ? (
          <p className="error">Error loading neutrino data</p>
        ) : (
          <div className="neutrino-details">
            <div className="neutrino-grid">
              <div className="neutrino-card">
                <h4>Mass Hierarchy</h4>
                <span className="highlight">{neutrinoQuery.data?.value?.hierarchy}</span>
                <p className="note falsifiable">
                  🔬 Inverted hierarchy would falsify IRH
                </p>
              </div>
              
              <div className="neutrino-card">
                <h4>Sum of Masses</h4>
                <span className="value">
                  Σm_ν = {neutrinoQuery.data?.value?.sum_masses_eV?.toFixed(4)} eV
                </span>
                <span className={neutrinoQuery.data?.details?.within_bound ? 'pass' : 'warning'}>
                  {neutrinoQuery.data?.details?.within_bound 
                    ? '✓ Within cosmological bound' 
                    : '⚠ Check bounds'}
                </span>
              </div>
              
              <div className="neutrino-card">
                <h4>Dirac/Majorana</h4>
                <span className="highlight">{neutrinoQuery.data?.details?.nature}</span>
                <p>Lepton number violation predicted</p>
              </div>
            </div>
            
            <div className="mass-eigenvalues">
              <h4>Mass Eigenvalues</h4>
              <table>
                <thead>
                  <tr>
                    <th>Eigenstate</th>
                    <th>Mass (eV)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>m₁</td>
                    <td>{neutrinoQuery.data?.value?.m1_eV?.toExponential(3)}</td>
                  </tr>
                  <tr>
                    <td>m₂</td>
                    <td>{neutrinoQuery.data?.value?.m2_eV?.toExponential(3)}</td>
                  </tr>
                  <tr>
                    <td>m₃</td>
                    <td>{neutrinoQuery.data?.value?.m3_eV?.toExponential(3)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="fermion-masses-note">
        <h2>Note on Fermion Masses</h2>
        <div className="note-box warning">
          <h4>Current Status of Fermion Mass Predictions</h4>
          <p>
            The IRH framework derives fermion masses from topological complexity K_f 
            values (Eq. 3.6). However, computational testing in notebooks has revealed 
            significant deviations from experimental values:
          </p>
          <ul>
            <li><strong>Electron:</strong> ~1700% deviation (largest)</li>
            <li><strong>First generation:</strong> ~613% average deviation</li>
            <li><strong>Second generation:</strong> ~72% average deviation (best)</li>
            <li><strong>Third generation:</strong> ~89% average deviation</li>
          </ul>
          <p>
            This indicates that the current K_f derivation methodology requires 
            refinement. The theoretical framework may need additional mechanisms 
            or modified coupling constants. See notebook 03_observable_extraction.ipynb 
            for detailed analysis.
          </p>
        </div>
      </section>
    </div>
  );
}

export default StandardModel;
