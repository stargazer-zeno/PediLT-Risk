// API base. Empty string = same origin (served behind nginx / vite proxy).
const BASE = import.meta.env.VITE_API_BASE || "";

async function asJson(resp) {
  const text = await resp.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text };
  }
  if (!resp.ok) {
    throw new Error(data.detail || `请求失败 (${resp.status})`);
  }
  return data;
}

export async function getHealth() {
  return asJson(await fetch(`${BASE}/api/health`));
}

export async function predictOne(patient) {
  return asJson(
    await fetch(`${BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient),
    })
  );
}

export async function createJob(file) {
  const form = new FormData();
  form.append("file", file);
  return asJson(await fetch(`${BASE}/api/jobs`, { method: "POST", body: form }));
}

export async function getJob(jobId) {
  return asJson(await fetch(`${BASE}/api/jobs/${jobId}`));
}

export async function getJobResults(jobId) {
  return asJson(await fetch(`${BASE}/api/jobs/${jobId}/results`));
}

export function downloadUrl(jobId, format) {
  return `${BASE}/api/jobs/${jobId}/download?format=${format}`;
}
