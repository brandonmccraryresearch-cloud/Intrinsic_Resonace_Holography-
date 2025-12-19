import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api.js';

/**
 * Falsification Component
 * 
 * Displays all falsifiable predictions with experimental tests
 * 
 * THEORETICAL REFERENCE: IRH v21.1 §7, Appendix J
 */
function Falsification() {
  const query = useQuery({
    queryKey: ['falsification'],
    queryFn: apiClient.getFalsificationSummary,
  });

  return (
    <div className="falsification-page">
      <header className="page-header">
        <h1>Falsifiable Predictions</h1>
        <p className="reference">IRH v21.1 §7, Appendix J</p>
      </header>

      <section className="intro">
        <h2>Scientific Method & IRH</h2>
        <p>
          IRH makes <strong>specific, quantitative predictions</strong> that can be 
          tested by near-term experiments. Unlike unfalsifiable theories, IRH clearly 
          states what observations would disprove it.
        </p>
      </section>

      <section className="predictions-table">
        <h2>Testable Predictions</h2>
        {query.isLoading ? (
          <p>Loading predictions...</p>
        ) : query.isError ? (
          <p className="error">Error loading data</p>
        ) : (
          <div className="predictions-grid">
            {query.data?.predictions?.map((pred, index) => (
              <div key={index} className="prediction-card">
                <div className="pred-header">
                  <h3>{pred.name}</h3>
                </div>
                
                <div className="pred-body">
                  <div className="pred-value">
                    <span className="label">IRH Prediction</span>
                    <span className="value">
                      {typeof pred.irh_value === 'number' 
                        ? pred.irh_value.toExponential(3) 
                        : pred.irh_value}
                    </span>
                  </div>
                  
                  <div className="falsification">
                    <span className="label">Would be falsified if:</span>
                    <span className="condition">{pred.falsification_condition}</span>
                  </div>
                  
                  <div className="experiments">
                    <span className="label">Test Experiments:</span>
                    <span className="exp-list">{pred.experiments?.join(', ')}</span>
                  </div>
                  
                  <div className="timeline">
                    <span className="label">Timeline:</span>
                    <span className="year">{pred.timeline}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="falsification-summary">
        <h2>Summary: What Would Falsify IRH?</h2>
        <div className="falsification-list">
          <div className="falsify-item danger">
            <h4>❌ Dark Energy w₀ = -1.00 ± 0.01</h4>
            <p>If Euclid/Roman confirm exact ΛCDM, IRH is falsified</p>
          </div>
          
          <div className="falsify-item danger">
            <h4>❌ Inverted Neutrino Hierarchy</h4>
            <p>If JUNO/DUNE confirm inverted hierarchy, IRH is falsified</p>
          </div>
          
          <div className="falsify-item danger">
            <h4>❌ LIV Parameter ξ &lt; 10⁻⁵</h4>
            <p>If CTA establishes stronger LIV bounds, IRH is falsified</p>
          </div>
          
          <div className="falsify-item danger">
            <h4>❌ Fourth Fermion Generation</h4>
            <p>If a fourth generation is discovered, n_inst = 3 is falsified</p>
          </div>
        </div>
      </section>

      <section className="timeline-section">
        <h2>Experimental Timeline</h2>
        <div className="timeline-visual">
          <div className="timeline-line"></div>
          <div className="timeline-events">
            <div className="event" style={{ left: '0%' }}>
              <span className="year">2025</span>
              <span className="name">Muon g-2 (FNAL)</span>
            </div>
            <div className="event" style={{ left: '25%' }}>
              <span className="year">2027</span>
              <span className="name">CMB-S4</span>
            </div>
            <div className="event" style={{ left: '45%' }}>
              <span className="year">2028</span>
              <span className="name">JUNO</span>
            </div>
            <div className="event critical" style={{ left: '55%' }}>
              <span className="year">2029</span>
              <span className="name">Euclid/CTA</span>
            </div>
            <div className="event" style={{ left: '75%' }}>
              <span className="year">2030</span>
              <span className="name">HL-LHC</span>
            </div>
            <div className="event" style={{ left: '95%' }}>
              <span className="year">2032</span>
              <span className="name">Einstein Tel.</span>
            </div>
          </div>
        </div>
        <p className="timeline-note">
          <strong>2029 is critical:</strong> Multiple experiments (Euclid, CTA) 
          will provide definitive tests of IRH predictions.
        </p>
      </section>

      <section className="philosophy">
        <h2>Falsifiability as Strength</h2>
        <blockquote>
          "The criterion of the scientific status of a theory is its falsifiability, 
          or refutability, or testability." — Karl Popper
        </blockquote>
        <p>
          IRH embraces falsifiability as a core feature. By making specific, testable 
          predictions that could be wrong, IRH demonstrates scientific integrity. 
          The theory either survives experimental tests or is replaced by something better.
        </p>
      </section>
    </div>
  );
}

export default Falsification;
