"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import ResultDialog from "@/components/ResultDialog";
import { ApiError, predict, type PredictionResult } from "@/lib/api";
import { addRecord, DEFAULT_PROFILE, makeThumb, PROFILE_KEY, useStored, type HistoryRecord } from "@/lib/store";

const OK_TYPES = ["image/jpeg", "image/png", "image/bmp"];
const OK_EXTS = [".jpg", ".jpeg", ".png", ".bmp"];
const ACCEPT_ATTR = "image/jpeg,image/png,image/bmp,.jpg,.jpeg,.png,.bmp";
const MAX_MB = 10;

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isValidImage(f: File) {
  const typeMatch = OK_TYPES.includes(f.type.toLowerCase());
  const nameMatch = OK_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext));
  return typeMatch || nameMatch;
}

const pct = (n?: number | null) => (typeof n === "number" ? `${(n * 100).toFixed(1)}%` : "-");

export default function Analisis() {
  const [profile] = useStored(PROFILE_KEY, DEFAULT_PROFILE);
  const input = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [rejected, setRejected] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [record, setRecord] = useState<HistoryRecord | null>(null);
  const [saved, setSaved] = useState(false);

  // Form patient fields
  const [patientId, setPatientId] = useState("");
  const [age, setAge] = useState<number | "">("");
  const [sex, setSex] = useState("Laki-laki");
  const [notes, setNotes] = useState("");

  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [url]);

  function pick(f?: File) {
    if (!f) return;
    setError("");
    setRejected("");
    setSaved(false);
    setResult(null);

    if (!isValidImage(f)) {
      return setError("Format berkas harus JPG, PNG, atau BMP.");
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      return setError(`Ukuran berkas (${formatSize(f.size)}) melebihi batas maksimal ${MAX_MB} MB.`);
    }

    if (url) URL.revokeObjectURL(url);
    setFile(f);
    setUrl(URL.createObjectURL(f));
  }

  function handleReset() {
    if (url) URL.revokeObjectURL(url);
    setFile(null);
    setUrl("");
    setError("");
    setRejected("");
    setResult(null);
    setRecord(null);
    setSaved(false);
    if (input.current) input.current.value = "";
  }

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!file) return setError("Pilih citra endoskopi terlebih dahulu.");

    setBusy(true);
    setError("");
    setRejected("");
    setSaved(false);
    setResult(null);

    try {
      // 1. Kirim file via FormData ke POST /predict backend
      const [predictionRes, thumb] = await Promise.all([predict(file), makeThumb(file)]);

      setResult(predictionRes);

      // Siapkan history record
      const rec: HistoryRecord = {
        id: crypto.randomUUID(),
        patientId: patientId || "Anonim",
        age: typeof age === "number" ? age : 0,
        sex,
        notes,
        at: new Date().toISOString(),
        file: file.name,
        thumb,
        label: predictionRes.label,
        confidence: predictionRes.confidence,
        probabilities: predictionRes.probabilities,
        gate_status: predictionRes.gate_status,
        original_image: predictionRes.original_image,
        heatmap_image: predictionRes.heatmap_image,
        overlay_image: predictionRes.overlay_image,
        model_name: predictionRes.model_name,
      };
      setRecord(rec);

      // Auto scroll ke hasil analisis
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 422) {
          // Ditolak oleh Image Type Gate
          setRejected(err.message);
        } else if (err.status === 0) {
          // Backend tidak terhubung
          setError("Backend tidak tersedia. Pastikan server FastAPI aktif di port 8000.");
        } else {
          setError(err.message);
        }
      } else {
        setError(err instanceof Error ? err.message : "Terjadi kesalahan saat analisis citra.");
      }
    } finally {
      setBusy(false);
    }
  }

  function handleSaveHistory() {
    if (record) {
      addRecord(record);
      setSaved(true);
    }
  }

  const isGerd = result?.predicted_class === "GERD";

  return (
    <div className="wrap" style={{ padding: "48px var(--pad) 80px" }}>
      <p className="eyebrow">Modul Analisis & Keterbukaan AI</p>
      <h1 style={{ fontSize: "clamp(28px,4vw,40px)", marginTop: 12 }}>
        Analisis Citra Endoskopi & Grad-CAM
      </h1>
      <p className="muted" style={{ marginTop: 12, maxWidth: "68ch" }}>
        Unggah citra endoskopi esofagus. Sistem akan memvalidasi tipe citra melalui Image Type Gate (224×224), mengklasifikasikan GERD dengan EfficientNetB4 (380×380), dan menghasilkan peta perhatian visual Grad-CAM.
      </p>

      {/* Form Input Citra & Pasien */}
      <form className="grid-2" style={{ marginTop: 40 }} onSubmit={submit}>
        {/* Kolom 1: Upload Citra */}
        <section className="panel stack" style={{ ["--gap" as string]: "16px" }}>
          <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
            <h3>1. Citra Endoskopi</h3>
            <span className="small muted">JPG, PNG, BMP · Maks {MAX_MB} MB</span>
          </div>

          {url ? (
            <div className="stack" style={{ ["--gap" as string]: "12px" }}>
              <div style={{ position: "relative", overflow: "hidden", borderRadius: "var(--r-md)", background: "#000" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={url} alt="Pratinjau citra endoskopi" className="preview" />
              </div>
              <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ minWidth: 0, maxWidth: "60%" }}>
                  <div className="small mono" style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {file?.name}
                  </div>
                  <span className="small muted">{file ? formatSize(file.size) : ""} · Siap dianalisis</span>
                </div>
                <div className="row" style={{ gap: 8 }}>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ padding: "6px 12px", minHeight: 34, fontSize: 13 }}
                    onClick={() => {
                      if (input.current) input.current.value = "";
                      input.current?.click();
                    }}
                  >
                    Ganti
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ padding: "6px 12px", minHeight: 34, fontSize: 13, color: "var(--danger)" }}
                    onClick={handleReset}
                  >
                    Hapus
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="stack" style={{ ["--gap" as string]: "12px" }}>
              <div
                role="button"
                tabIndex={0}
                className={`drop ${over ? "over" : ""}`}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 12,
                  cursor: "pointer",
                  outline: "none",
                  transition: "all 0.15s ease",
                  padding: "40px 20px",
                }}
                onClick={() => {
                  if (input.current) input.current.value = "";
                  input.current?.click();
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    if (input.current) input.current.value = "";
                    input.current?.click();
                  }
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  setOver(true);
                }}
                onDragLeave={() => setOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setOver(false);
                  pick(e.dataTransfer.files?.[0]);
                }}
              >
                <div
                  style={{
                    width: 52,
                    height: 52,
                    borderRadius: "50%",
                    background: "var(--surface)",
                    display: "grid",
                    placeItems: "center",
                    color: "var(--primary)",
                  }}
                >
                  <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <div>
                  <p style={{ fontWeight: 600, fontSize: 15 }}>Tarik citra ke sini atau klik untuk memilih berkas</p>
                  <p className="hint" style={{ marginTop: 4 }}>JPG, PNG, atau BMP · maksimal {MAX_MB} MB</p>
                </div>
              </div>
            </div>
          )}

          <input
            ref={input}
            id="image-file-input"
            type="file"
            accept={ACCEPT_ATTR}
            hidden
            onChange={(e) => {
              pick(e.target.files?.[0]);
              e.target.value = "";
            }}
            aria-label="Pilih citra endoskopi"
          />

          {/* Alert Penolakan Image Gate */}
          {rejected && (
            <div className="notice bad" role="alert" style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
              <span style={{ fontSize: 20, lineHeight: 1 }}>🛡️</span>
              <div>
                <b>Citra ditolak oleh Image Type Gate:</b>
                <p style={{ marginTop: 4 }}>{rejected}</p>
                <p className="small muted" style={{ marginTop: 6 }}>
                  Sistem keamanan Image Type Gate mendeteksi gambar bukan hasil endoskopi esofagus/lambung. Model EfficientNetB4 dan visualisasi Grad-CAM tidak dijalankan.
                </p>
              </div>
            </div>
          )}

          {/* Alert Error Umum / Backend Mati */}
          {error && (
            <div className="notice bad" role="alert" style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
              <span style={{ fontSize: 20, lineHeight: 1 }}>⚠️</span>
              <div>
                <b>Gagal Memproses Citra:</b>
                <p style={{ marginTop: 4 }}>{error}</p>
              </div>
            </div>
          )}
        </section>

        {/* Kolom 2: Informasi Pasien & Tombol Submit */}
        <section className="panel stack" style={{ ["--gap" as string]: "16px" }}>
          <h3>2. Informasi Pasien & Analisis</h3>
          <div className="field">
            <label htmlFor="patientId">ID Pasien Anonim *</label>
            <input
              id="patientId"
              name="patientId"
              className="input"
              required
              placeholder="PS-8821"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
            />
            <span className="hint">Gunakan kode anonim untuk privasi medis.</span>
          </div>

          <div className="grid-2" style={{ gap: 16 }}>
            <div className="field">
              <label htmlFor="age">Usia (tahun) *</label>
              <input
                id="age"
                name="age"
                type="number"
                min={1}
                max={120}
                className="input"
                required
                value={age}
                onChange={(e) => setAge(e.target.value === "" ? "" : Number(e.target.value))}
              />
            </div>
            <div className="field">
              <span className="label">Jenis Kelamin *</span>
              <div className="seg">
                {["Laki-laki", "Perempuan", "Lainnya"].map((s) => (
                  <label key={s}>
                    <input
                      type="radio"
                      name="sex"
                      value={s}
                      checked={sex === s}
                      onChange={() => setSex(s)}
                    />
                    {s}
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div className="field">
            <label htmlFor="notes">Catatan Indikasi Klinis (opsional)</label>
            <textarea
              id="notes"
              name="notes"
              className="textarea"
              placeholder="Keluhan heartburn, disfagia, riwayat esofagitis..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: "100%", padding: "12px 20px", fontSize: 15, marginTop: 8 }}
            disabled={busy || !file}
          >
            {busy ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
                <span
                  style={{
                    width: 16,
                    height: 16,
                    border: "2px solid #fff",
                    borderTopColor: "transparent",
                    borderRadius: "50%",
                    display: "inline-block",
                    animation: "spin 0.8s linear infinite",
                  }}
                />
                Menganalisis citra & Grad-CAM…
              </span>
            ) : (
              "Mulai Analisis Medis"
            )}
          </button>

          {busy && (
            <p className="small muted" style={{ textAlign: "center", marginTop: 4 }}>
              Memvalidasi gate (224×224), inferensi EfficientNetB4 (380×380), dan menghitung aktivasi Grad-CAM...
            </p>
          )}
        </section>
      </form>

      {/* ================================================================= */}
      {/* BAGIAN HASIL ANALISIS & VISUALISASI GRAD-CAM */}
      {/* ================================================================= */}
      {result && (
        <section
          ref={resultsRef}
          className="panel"
          style={{
            marginTop: 48,
            border: `2px solid ${isGerd ? "var(--warn)" : "var(--ok)"}`,
            boxShadow: "0 10px 30px -10px rgba(0,0,0,0.1)",
            padding: "32px var(--pad)",
          }}
        >
          {/* 1. Status Analisis */}
          <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", borderBottom: "1px solid var(--line)", paddingBottom: 16 }}>
            <div>
              <div className="row" style={{ gap: 8, marginBottom: 6 }}>
                <span className="badge" style={{ background: "var(--ok-bg)", color: "var(--ok)", fontSize: 12, padding: "4px 10px" }}>
                  ✓ Status: Analisis Selesai
                </span>
                <span className="badge" style={{ background: "var(--surface-2)", color: "var(--primary)", fontSize: 12, padding: "4px 10px" }}>
                  Gate: {result.gate_status}
                </span>
              </div>
              <h2 style={{ fontSize: "clamp(24px, 3.5vw, 32px)", marginTop: 4 }}>
                Hasil Analisis Medis
              </h2>
              <p className="small muted mono" style={{ marginTop: 4 }}>
                Model: {result.model_name} · Gate Score: {result.gate_score ?? "< 0.01"}
              </p>
            </div>

            <div className="row" style={{ gap: 10 }}>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleReset}
                style={{ fontSize: 13, padding: "8px 14px", minHeight: 36 }}
              >
                + Unggah Gambar Baru
              </button>
            </div>
          </div>

          {/* 2 & 3 & 4. Prediksi Kelas, Probabilitas, dan Confidence */}
          <div className="grid-2" style={{ marginTop: 24, gap: 24 }}>
            {/* Box Prediksi Utama */}
            <div
              style={{
                background: isGerd ? "var(--warn-bg)" : "var(--ok-bg)",
                borderRadius: "var(--r-lg)",
                padding: "24px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}
            >
              <div>
                <span className="eyebrow" style={{ color: isGerd ? "var(--warn)" : "var(--ok)" }}>
                  Hasil Klasifikasi Model
                </span>
                <div
                  style={{
                    fontSize: "clamp(36px, 5vw, 48px)",
                    fontWeight: 800,
                    letterSpacing: "-0.02em",
                    color: isGerd ? "var(--warn)" : "var(--ok)",
                    marginTop: 8,
                  }}
                >
                  {result.predicted_class}
                </div>
                <p style={{ marginTop: 8, fontSize: 14, color: "var(--ink)", opacity: 0.9 }}>
                  {isGerd
                    ? "Citra terdeteksi memiliki karakteristik visual sesuai Gastroesophageal Reflux Disease (GERD)."
                    : "Citra terdeteksi dalam batas normal (GERD_Normal) tanpa indikasi lesi refluks esofagus yang signifikan."}
                </p>
              </div>

              <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid rgba(0,0,0,0.08)" }}>
                <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ fontSize: 13, fontWeight: 600 }}>Tingkat Keyakinan (Confidence):</span>
                  <span className="mono" style={{ fontSize: 22, fontWeight: 700, color: isGerd ? "var(--warn)" : "var(--ok)" }}>
                    {pct(result.confidence)}
                  </span>
                </div>
                <div className={`bar ${isGerd ? "warn" : ""}`} style={{ marginTop: 8, height: 10 }}>
                  <i style={{ width: pct(result.confidence) }} />
                </div>
              </div>
            </div>

            {/* Box Probabilitas Kedua Kelas */}
            <div
              style={{
                background: "var(--surface)",
                borderRadius: "var(--r-lg)",
                padding: "24px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}
            >
              <div>
                <span className="eyebrow">Distribusi Probabilitas Kelas</span>
                <p className="small muted" style={{ marginTop: 4 }}>
                  Nilai probabilitas keluaran sigmoid Dense neuron EfficientNetB4:
                </p>

                <div className="stack" style={{ ["--gap" as string]: "16px", marginTop: 20 }}>
                  {/* Probabilitas GERD */}
                  <div>
                    <div className="row small" style={{ justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ fontWeight: 600 }}>P(GERD) - Kelas 1</span>
                      <span className="mono" style={{ fontWeight: 700, color: "var(--warn)" }}>
                        {pct(result.probability_gerd)}
                      </span>
                    </div>
                    <div className="bar warn" style={{ height: 10 }}>
                      <i style={{ width: pct(result.probability_gerd) }} />
                    </div>
                  </div>

                  {/* Probabilitas GERD_Normal */}
                  <div>
                    <div className="row small" style={{ justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ fontWeight: 600 }}>P(GERD_Normal) - Kelas 0</span>
                      <span className="mono" style={{ fontWeight: 700, color: "var(--ok)" }}>
                        {pct(result.probability_normal)}
                      </span>
                    </div>
                    <div className="bar" style={{ height: 10 }}>
                      <i style={{ width: pct(result.probability_normal) }} />
                    </div>
                  </div>
                </div>
              </div>

              <div className="small muted" style={{ marginTop: 20, paddingTop: 12, borderTop: "1px solid var(--line)" }}>
                Pasien: <b className="mono">{patientId || "Anonim"}</b> · Usia: <b>{age ? `${age} th` : "-"}</b> · Jenis kelamin: <b>{sex}</b>
              </div>
            </div>
          </div>

          {/* 5, 6, 7 & 8. Visualisasi Gambar: Asli, Grad-CAM Heatmap, dan Grad-CAM Overlay */}
          <div style={{ marginTop: 36 }}>
            <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
              <div>
                <h3 style={{ fontSize: 20 }}>Visualisasi Keterbukaan Model (Grad-CAM)</h3>
                <p className="small muted" style={{ marginTop: 4 }}>
                  Layer target: <code className="mono">top_conv</code> (EfficientNetB4) · Resolusi fitur: 12 × 12 × 1792
                </p>
              </div>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                gap: 20,
                marginTop: 16,
              }}
            >
              {/* Gambar 1: Citra Asli */}
              <div
                style={{
                  background: "var(--bg)",
                  border: "1px solid var(--line)",
                  borderRadius: "var(--r-lg)",
                  overflow: "hidden",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontWeight: 600, fontSize: 13 }}>1. Citra Asli</span>
                  <span className="small muted">Endoskopi</span>
                </div>
                <div style={{ background: "#000", padding: 8, display: "grid", placeItems: "center", minHeight: 240, flex: 1 }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={result.original_image || url}
                    alt="Citra endoskopi asli"
                    style={{ width: "100%", maxHeight: 280, objectFit: "contain", display: "block", borderRadius: "var(--r-sm)" }}
                  />
                </div>
                <div className="small muted" style={{ padding: "8px 12px", background: "var(--surface)" }}>
                  Resolusi asli masukan klinis
                </div>
              </div>

              {/* Gambar 2: Heatmap Grad-CAM */}
              {result.heatmap_image ? (
                <div
                  style={{
                    background: "var(--bg)",
                    border: "1px solid var(--line)",
                    borderRadius: "var(--r-lg)",
                    overflow: "hidden",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: "var(--primary)" }}>2. Grad-CAM Heatmap</span>
                    <span className="small badge" style={{ background: "var(--surface-2)" }}>JET Colormap</span>
                  </div>
                  <div style={{ background: "#000", padding: 8, display: "grid", placeItems: "center", minHeight: 240, flex: 1 }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={result.heatmap_image}
                      alt="Grad-CAM Heatmap"
                      style={{ width: "100%", maxHeight: 280, objectFit: "contain", display: "block", borderRadius: "var(--r-sm)" }}
                    />
                  </div>
                  <div className="small muted" style={{ padding: "8px 12px", background: "var(--surface)" }}>
                    Peta aktivasi gradien layer konvolusi
                  </div>
                </div>
              ) : null}

              {/* Gambar 3: Overlay Grad-CAM */}
              {result.overlay_image ? (
                <div
                  style={{
                    background: "var(--bg)",
                    border: "1px solid var(--line)",
                    borderRadius: "var(--r-lg)",
                    overflow: "hidden",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: isGerd ? "var(--warn)" : "var(--ok)" }}>
                      3. Grad-CAM Overlay
                    </span>
                    <span className="small badge" style={{ background: "var(--surface-2)" }}>Alpha 45%</span>
                  </div>
                  <div style={{ background: "#000", padding: 8, display: "grid", placeItems: "center", minHeight: 240, flex: 1 }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={result.overlay_image}
                      alt="Grad-CAM Overlay pada Citra Asli"
                      style={{ width: "100%", maxHeight: 280, objectFit: "contain", display: "block", borderRadius: "var(--r-sm)" }}
                    />
                  </div>
                  <div className="small muted" style={{ padding: "8px 12px", background: "var(--surface)" }}>
                    Superimposisi visualisasi fokus model
                  </div>
                </div>
              ) : null}
            </div>
          </div>

          {/* Penjelasan Edukasi Grad-CAM Sesuai Ketentuan */}
          <div
            className="notice"
            style={{
              marginTop: 24,
              background: "var(--surface-2)",
              border: "1px solid var(--line)",
              display: "flex",
              gap: 12,
              alignItems: "flex-start",
            }}
          >
            <span style={{ fontSize: 20, lineHeight: 1 }}>💡</span>
            <div style={{ fontSize: 13, lineHeight: 1.6 }}>
              <b>Penjelasan Interpretasi Visual Grad-CAM:</b>
              <p style={{ marginTop: 4 }}>
                Grad-CAM (Gradient-weighted Class Activation Mapping) menunjukkan area atau pola visual pada citra yang <b>paling memengaruhi keputusan prediksi model EfficientNetB4</b> (region berbobot gradien tinggi ditandai warna merah/kuning).
              </p>
              <p style={{ marginTop: 4, color: "var(--muted)" }}>
                Peta perhatian ini berfungsi sebagai instrumen keterbukaan (explainable AI) dan <b>bukan lokasi lesi patologis yang telah terdiagnosis definitif</b>. Penegakan diagnosis dan evaluasi medis wajib dilakukan oleh dokter spesialis.
              </p>
            </div>
          </div>

          {/* Tombol Aksi di Bawah Hasil */}
          <div
            className="row no-print"
            style={{
              marginTop: 32,
              justifyContent: "space-between",
              borderTop: "1px solid var(--line)",
              paddingTop: 20,
            }}
          >
            <div className="row" style={{ gap: 10 }}>
              <button type="button" className="btn btn-ghost" onClick={() => window.print()}>
                🖨️ Cetak Hasil
              </button>
              {record && !saved && (
                <button type="button" className="btn btn-primary" onClick={handleSaveHistory}>
                  Simpan ke Riwayat
                </button>
              )}
              {saved && (
                <span className="badge normal" style={{ padding: "8px 12px" }}>
                  ✓ Tersimpan di Riwayat (<Link href="/riwayat">Lihat Riwayat</Link>)
                </span>
              )}
            </div>

            <button
              type="button"
              className="btn btn-ghost"
              onClick={handleReset}
              style={{ fontWeight: 600 }}
            >
              Unggah Citra Baru →
            </button>
          </div>
        </section>
      )}

      {/* Catatan Legal & Tanggung Jawab */}
      <p className="notice warn" style={{ marginTop: 40 }}>
        <b>Batas tanggung jawab klinis:</b> Hasil klasifikasi dan peta Grad-CAM membantu tenaga medis menyaring kasus endoskopi dan bukan alat diagnosis mandiri. Keputusan terapi medis tetap ditetapkan oleh dokter spesialis gastroenterologi.
      </p>

      {/* Modal Dialog Detail Pasien jika dibutuhkan */}
      {record && saved && (
        <ResultDialog
          record={record}
          threshold={profile.threshold}
          onClose={() => setRecord(null)}
        />
      )}
    </div>
  );
}
