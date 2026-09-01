import React from "react";

function Dot({ ok }) {
  return <span className={`dot ${ok ? "dot-ok" : "dot-bad"}`} />;
}

export default function HealthBar({ health, error, onRefresh }) {
  const ml = health?.ml || {};
  const llm = health?.llm || {};
  return (
    <div className="healthbar">
      <div className="health-item">
        <Dot ok={!!health && !error} /> Backend
      </div>
      <div className="health-item">
        <Dot ok={!!ml.available} /> XGBoost ML
        {ml.error && <span className="health-note" title={ml.error}>(Unavailable)</span>}
      </div>
      <div className="health-item">
        <Dot ok={!!llm.enabled} /> LLM
        {!llm.enabled && <span className="health-note">(Not configured)</span>}
        {llm.enabled && llm.reachable === false && (
          <span className="health-note">(Unreachable)</span>
        )}
      </div>
      <button className="btn-small" onClick={onRefresh}>
        Refresh status
      </button>
      {error && <span className="health-error">{error}</span>}
    </div>
  );
}
