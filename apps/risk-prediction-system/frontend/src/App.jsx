import React, { useCallback, useEffect, useRef, useState } from "react";
import HealthBar from "./components/HealthBar.jsx";
import ResultsTable from "./components/ResultsTable.jsx";
import {
  createJob,
  downloadUrl,
  getHealth,
  getJob,
  getJobResults,
  predictOne,
} from "./api.js";

const SAMPLE_URL = "/samples/demo_long_followup.json";

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");

  const [jsonText, setJsonText] = useState("");
  const [singleNotice, setSingleNotice] = useState("");
  const [singleResult, setSingleResult] = useState(null);
  const [singleError, setSingleError] = useState("");
  const [singleBusy, setSingleBusy] = useState(false);
  const jsonInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchError, setBatchError] = useState("");
  const [batchBusy, setBatchBusy] = useState(false);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await getHealth());
      setHealthError("");
    } catch (e) {
      setHealthError(e.message);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  async function handleSingle() {
    if (!jsonText.trim()) {
      setSingleNotice("");
      setSingleError("请粘贴标准患者 JSON，或点击“载入示例”。");
      jsonInputRef.current?.focus();
      return;
    }
    setSingleBusy(true);
    setSingleNotice("");
    setSingleError("");
    setSingleResult(null);
    try {
      const patient = JSON.parse(jsonText);
      const res = await predictOne(patient);
      setSingleResult(res);
    } catch (e) {
      setSingleError(e.message);
    } finally {
      setSingleBusy(false);
    }
  }

  async function handleLoadSample() {
    setSingleNotice("");
    setSingleError("");
    try {
      const response = await fetch(SAMPLE_URL, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const sample = await response.json();
      setJsonText(JSON.stringify(sample, null, 2));
    } catch (e) {
      setSingleError(`示例数据加载失败：${e.message}`);
      return;
    }
    setSingleResult(null);
    setSingleNotice(
      "长期随访模拟样例已载入。该数据为合成数据，仅用于展示长期随访处理和模型链路，可以直接提交预测。"
    );
    requestAnimationFrame(() => {
      jsonInputRef.current?.focus();
      jsonInputRef.current?.setSelectionRange(0, 0);
    });
  }

  function handleClearSingle() {
    setJsonText("");
    setSingleResult(null);
    setSingleError("");
    setSingleNotice("");
    jsonInputRef.current?.focus();
  }

  // Poll a job until it finishes.
  useEffect(() => {
    if (!job || ["completed", "failed"].includes(job.status)) return;
    const timer = setInterval(async () => {
      try {
        const meta = await getJob(job.job_id);
        setJob(meta);
        if (meta.status === "completed") {
          const r = await getJobResults(job.job_id);
          setBatchResults(r.results);
        }
      } catch (e) {
        setBatchError(e.message);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [job]);

  async function handleUpload() {
    if (!file) {
      setBatchError("请先选择文件。");
      return;
    }
    setBatchBusy(true);
    setBatchError("");
    setBatchResults(null);
    setJob(null);
    try {
      const meta = await createJob(file);
      setJob(meta);
    } catch (e) {
      setBatchError(e.message);
    } finally {
      setBatchBusy(false);
    }
  }

  const singleRows = singleResult ? [singleResult] : null;
  const progress =
    job?.total > 0 ? Math.min(100, Math.round((job.completed / job.total) * 100)) : 0;

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-copy">
          <div className="eyebrow">PEDILT-RISK · RESEARCH DEMO</div>
          <h1>小儿肝移植死亡风险预测</h1>
          <p className="subtitle">
            结合 XGBoost 与临床大模型，评估未来 1 个月、1 年及 5 年死亡风险
          </p>
          <div className="hero-tags">
            <span>双模型并列评估</span>
            <span>临床 Pattern 解析</span>
            <span>批量任务支持</span>
          </div>
        </div>
        <div className="hero-mark" aria-hidden="true">
          <span>AI</span>
          <small>Risk Assistant</small>
        </div>
      </header>

      <HealthBar health={health} error={healthError} onRefresh={refreshHealth} />

      <main>
        <section className="card workspace-card">
          <div className="section-heading">
            <div className="section-number">01</div>
            <div>
              <h2>单患者预测</h2>
              <p>粘贴一条标准患者 JSON，系统将同步调用 ML 与 LLM 两条预测链路。</p>
            </div>
          </div>

          <div className="editor-shell">
            <div className="editor-toolbar">
              <span className="editor-title">患者数据 · JSON</span>
              <span className="editor-meta">
                {jsonText.trim() ? `${jsonText.length} 字符` : "等待输入"}
              </span>
            </div>
            <textarea
              ref={jsonInputRef}
              aria-label="患者标准 JSON"
              value={jsonText}
              onChange={(e) => {
                setJsonText(e.target.value);
                setSingleNotice("");
              }}
              rows={12}
              spellCheck={false}
              placeholder={'请在此粘贴标准患者 JSON，或点击下方“载入示例”…'}
            />
          </div>

          <div className="action-row">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSingle}
              disabled={singleBusy}
            >
              {singleBusy ? <span className="spinner" aria-hidden="true" /> : null}
              {singleBusy ? "模型分析中…" : "提交预测"}
            </button>
            <button type="button" className="btn btn-secondary" onClick={handleLoadSample}>
              载入长期随访模拟样例
            </button>
            {jsonText && (
              <button type="button" className="btn btn-text" onClick={handleClearSingle}>
                清空
              </button>
            )}
          </div>
          {singleNotice && <p className="notice success-notice">{singleNotice}</p>}
          {singleError && <p className="notice error-notice">{singleError}</p>}

          {singleResult && (
            <div className="result-section">
              <div className="result-heading">
                <span>预测结果</span>
                <small>ML 与 LLM 独立输出，不做加权融合</small>
              </div>
              <ResultsTable results={singleRows} />
            </div>
          )}
        </section>

        <section className="card workspace-card">
          <div className="section-heading">
            <div className="section-number">02</div>
            <div>
              <h2>批量预测</h2>
              <p>上传患者数据文件，后台异步执行并生成可下载的结构化结果。</p>
            </div>
          </div>

          <label className="upload-zone">
            <input
              type="file"
              accept=".json,.jsonl,.csv,.xlsx,.xls"
              onChange={(e) => {
                setFile(e.target.files[0] || null);
                setBatchError("");
              }}
            />
            <span className="upload-icon" aria-hidden="true">↑</span>
            <span className="upload-copy">
              <strong>{file ? file.name : "选择患者数据文件"}</strong>
              <small>
                {file
                  ? `${(file.size / 1024).toFixed(1)} KB`
                  : "支持 JSON、JSONL、CSV、XLSX，每位患儿一条记录"}
              </small>
            </span>
            <span className="upload-action">{file ? "重新选择" : "浏览文件"}</span>
          </label>

          <div className="action-row">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={batchBusy || !file}
            >
              {batchBusy ? <span className="spinner" aria-hidden="true" /> : null}
              {batchBusy ? "正在创建任务…" : "上传并预测"}
            </button>
            {file && <span className="selection-note">已选择：{file.name}</span>}
          </div>

          {job && (
            <div className="job-status">
              <div className="job-status-header">
                <div>
                  <span className="job-label">任务 ID</span>
                  <code>{job.job_id}</code>
                </div>
                <span className={`job-state job-state-${job.status}`}>{job.status}</span>
              </div>
              <div className="progress-info">
                <span>处理进度</span>
                <strong>{job.completed}/{job.total}</strong>
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-valuenow={progress}
                aria-valuemin="0"
                aria-valuemax="100"
              >
                <span style={{ width: `${progress}%` }} />
              </div>
              {job.failed ? <div className="failed-note">失败记录：{job.failed}</div> : null}
              {job.status === "completed" && (
                <div className="download-row">
                  <a className="btn btn-download" href={downloadUrl(job.job_id, "csv")}>
                    下载 CSV
                  </a>
                  <a className="btn btn-download" href={downloadUrl(job.job_id, "json")}>
                    下载 JSON
                  </a>
                </div>
              )}
            </div>
          )}
          {batchError && <p className="notice error-notice">{batchError}</p>}
          {batchResults && (
            <div className="result-section">
              <div className="result-heading">
                <span>批量预测结果</span>
                <small>共 {batchResults.length} 条记录</small>
              </div>
              <ResultsTable results={batchResults} />
            </div>
          )}
        </section>
      </main>

      <footer>
        <span>PediLT-Risk Prediction System</span>
        <span className="footer-divider">·</span>
        <span>仅供科研演示，不用于临床决策</span>
      </footer>
    </div>
  );
}
