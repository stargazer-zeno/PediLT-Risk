import React from "react";
import {
  displayRationale,
  formatProbability,
  normalizePatterns,
  resultStatusClass,
} from "../patterns.js";

function StatusTag({ status }) {
  return <span className={resultStatusClass(status)}>{status}</span>;
}

function hasDetails(llm) {
  return !!(
    llm?.patterns?.length ||
    llm?.pattern_output ||
    displayRationale(llm) ||
    llm?.parse_status ||
    llm?.parse_warnings?.length ||
    llm?.error
  );
}

export default function ResultsTable({ results }) {
  if (!results || results.length === 0) return null;
  return (
    <>
      <div className="table-wrap">
        <table className="results">
          <thead>
            <tr>
              <th rowSpan="2">ID</th>
              <th colSpan="4">XGBoost (ML)</th>
              <th colSpan="5">LLM / vLLM</th>
            </tr>
            <tr>
              <th>Status</th>
              <th>1 Month</th>
              <th>1 Year</th>
              <th>5 Years</th>
              <th>Status</th>
              <th>1 Month</th>
              <th>1 Year</th>
              <th>5 Years</th>
              <th>Pattern Count</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={r.id || i}>
                <td className="id-col">{r.id}</td>
                <td><StatusTag status={r.ml?.status} /></td>
                <td>{formatProbability(r.ml?.death_probability_1m)}</td>
                <td>{formatProbability(r.ml?.death_probability_1y)}</td>
                <td>{formatProbability(r.ml?.death_probability_5y)}</td>
                <td><StatusTag status={r.llm?.status} /></td>
                <td>{formatProbability(r.llm?.death_probability_1m, { showNull: r.llm?.status === "ok" })}</td>
                <td>{formatProbability(r.llm?.death_probability_1y, { showNull: r.llm?.status === "ok" })}</td>
                <td>{formatProbability(r.llm?.death_probability_5y, { showNull: r.llm?.status === "ok" })}</td>
                <td>{r.llm?.pair_count ?? r.llm?.patterns?.length ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="llm-detail-list">
        {results.filter((r) => hasDetails(r.llm)).map((r, i) => {
          const patterns = normalizePatterns(r.llm);
          const rationale = displayRationale(r.llm);
          return (
          <section className="llm-detail-card" key={`${r.id || i}-llm-detail`}>
            <div className="llm-detail-header">
              <div>
                <span className="detail-label">LLM Clinical Patterns</span>
                <span className="id-col">{r.id || `row_${i + 1}`}</span>
              </div>
              <div className="detail-statuses">
                <StatusTag status={r.llm?.status} />
                {r.llm?.parse_status && (
                  <span className="parse-status">
                    Parse status: {r.llm.parse_status}
                  </span>
                )}
              </div>
            </div>

            {patterns.length > 0 && (
              <div className="pattern-list">
                {patterns.map((item, idx) => (
                  <div className="pattern-item" key={`${r.id || i}-pattern-${idx}`}>
                    <div className="pattern-index">Pattern {idx + 1}</div>
                    <div className="pattern-content">
                      <div className="pattern-title">{item.pattern || "Pattern name unavailable"}</div>
                      {item.analysis && (
                        <div className="pattern-analysis">
                          <span className="analysis-label">Clinical Analysis</span>
                          {item.analysis}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {patterns.length === 0 && r.llm?.pattern_output && (
              <pre className="pattern-raw">{r.llm.pattern_output}</pre>
            )}

            {rationale && (
              <div className="answer-text">
                <strong>Overall Rationale:</strong>
                <span>{rationale}</span>
              </div>
            )}

            {r.llm?.parse_warnings?.length > 0 && (
              <div className="warnings">
                <strong>Parse Warnings:</strong>{r.llm.parse_warnings.join("; ")}
              </div>
            )}

            {r.llm?.error && <div className="error">LLM Error: {r.llm.error}</div>}
          </section>
          );
        })}
      </div>
    </>
  );
}
