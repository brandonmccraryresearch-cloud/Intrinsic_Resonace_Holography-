/**
 * API client for IRH backend
 * 
 * Connects to FastAPI backend at localhost:8000
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions for each endpoint
export const apiClient = {
  // Health check
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Fixed Point
  async getFixedPoint() {
    const response = await api.get('/api/v1/fixed-point');
    return response.data;
  },

  // RG Flow
  async computeRGFlow(params) {
    const response = await api.post('/api/v1/rg-flow', params);
    return response.data;
  },

  // Observables
  async getUniversalExponent() {
    const response = await api.get('/api/v1/observables/C_H');
    return response.data;
  },

  async getFineStructureConstant() {
    const response = await api.get('/api/v1/observables/alpha');
    return response.data;
  },

  async getDarkEnergyEOS() {
    const response = await api.get('/api/v1/observables/dark-energy');
    return response.data;
  },

  async getLIVParameter() {
    const response = await api.get('/api/v1/observables/liv');
    return response.data;
  },

  // Standard Model
  async getGaugeGroup() {
    const response = await api.get('/api/v1/standard-model/gauge-group');
    return response.data;
  },

  async getNeutrinoPredictions() {
    const response = await api.get('/api/v1/standard-model/neutrinos');
    return response.data;
  },

  // Falsification
  async getFalsificationSummary() {
    const response = await api.get('/api/v1/falsification/summary');
    return response.data;
  },
};

export default api;
