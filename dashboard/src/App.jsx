import { useState, useEffect } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const riskColors = {
  high: '#d63a3a',
  medium: '#d68a10',
  low: '#5a9e5a',
};

function App() {
  const [date, setDate] = useState('2015-06-01');
  const [machines, setMachines] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [diagnosis, setDiagnosis] = useState(null);
  const [diagnosing, setDiagnosing] = useState(false);

  const fetchMachines = async () => {
    setLoading(true);
    setSelectedMachine(null);
    setDiagnosis(null);
    try {
      const res = await fetch(API_BASE + '/machines-at-risk?date=' + date);
      const data = await res.json();
      setMachines(data.machines);
    } catch (err) {
      console.error('Failed to fetch machines:', err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchMachines();
  }, []);

  const runDiagnosis = async (machineId) => {
    setSelectedMachine(machineId);
    setDiagnosing(true);
    setDiagnosis(null);
    try {
      const res = await fetch(API_BASE + '/diagnose/' + machineId + '?date=' + date);
      const data = await res.json();
      setDiagnosis(data);
    } catch (err) {
      console.error('Diagnosis failed:', err);
    }
    setDiagnosing(false);
  };

  return (
    <div className="dashboard">
      <header>
        <h1>Agentic AI Predictive Maintenance</h1>
        <p className="subtitle">Validated on real historical failure data - zero simulated data</p>
      </header>

      <div className="controls">
        <label>
          Select date:
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        </label>
        <button onClick={fetchMachines} disabled={loading}>
          {loading ? 'Loading...' : 'Check machines'}
        </button>
      </div>

      <div className="main-grid">
        <div className="machine-list">
          <h3>Machines by risk ({machines.length})</h3>
          <div className="list-scroll">
            {machines.map((m) => {
              const rowClass = selectedMachine === m.machineId ? 'machine-row selected' : 'machine-row';
              return (
                <div
                  key={m.machineId}
                  className={rowClass}
                  onClick={() => runDiagnosis(m.machineId)}
                >
                  <span className="machine-id">Machine {m.machineId}</span>
                  <span className="health-score">{m.healthScore.toFixed(3)}</span>
                  <span className="risk-badge" style={{ backgroundColor: riskColors[m.riskLevel] }}>
                    {m.riskLevel}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="detail-panel">
          {!selectedMachine && <p className="placeholder">Click a machine to run full diagnosis</p>}
          {diagnosing && <p className="placeholder">Running agent pipeline...</p>}
          {diagnosis && !diagnosing && (
            <div>
              <h3>Machine {diagnosis.machineId} - Full Diagnosis</h3>
              <div className="metrics-row">
                <div className="metric">
                  <span className="metric-label">Health</span>
                  <span className="metric-value">{diagnosis.riskLevel}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Diagnosis</span>
                  <span className="metric-value">{diagnosis.diagnosis ? diagnosis.diagnosis : 'N/A'}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Confidence</span>
                  <span className="metric-value">{diagnosis.confidence ? diagnosis.confidence + '%' : 'N/A'}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Routing</span>
                  <span className="metric-value">{diagnosis.routing.replace('_', ' ')}</span>
                </div>
              </div>

              <div className="evidence-grid">
                <div className="evidence-box">
                  <h4>Reasoning</h4>
                  <p>{diagnosis.reasoning}</p>
                </div>
                <div className="evidence-box">
                  <h4>Recommendation</h4>
                  <p>{diagnosis.recommendation}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;