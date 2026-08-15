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
        <Dot ok={!!health && !error} /> 后端 Backend
      </div>
      <div className="health-item">
        <Dot ok={!!ml.available} /> XGBoost ML
        {ml.error && <span className="health-note" title={ml.error}>（不可用）</span>}
      </div>
      <div className="health-item">
        <Dot ok={!!llm.enabled} /> LLM
        {!llm.enabled && <span className="health-note">（未配置）</span>}
        {llm.enabled && llm.reachable === false && (
          <span className="health-note">（不可达）</span>
        )}
      </div>
      <button className="btn-small" onClick={onRefresh}>
        刷新状态
      </button>
      {error && <span className="health-error">{error}</span>}
    </div>
  );
}
