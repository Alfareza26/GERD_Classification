"use client";
import { useEffect, useRef } from "react";
import type { HistoryRecord } from "@/lib/store";

const pct = (n?: number | null) => (typeof n === "number" ? `${(n * 100).toFixed(1)}%` : "-");

type Props = {
  record: HistoryRecord;
  threshold: number;
  onClose: () => void;
  onSave?: () => void;
};

export default function ResultDialog({ record, threshold, onClose, onSave }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);

  const isGerd = record.label === "GERD";
  const low = record.confidence * 100 < threshold;

  return (
    <dialog ref={ref} onClose={onClose} aria-labelledby="dlg-title" style={{ maxWidth: 840, width: "95vw" }}>
      <div className="dlg-body" style={{ maxHeight: "85vh", overflowY: "auto" }}>
        {/* Header Dialog */}
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <p className="eyebrow" id="dlg-title">
              Hasil Analisis Medis
            </p>
            <p className="small muted mono" style={{ marginTop: 4 }}>
              Model: {record.model_name || "EfficientNetB4_GERD_Model"} · Gate: {record.gate_status || "LOLOS_GATE"}
            </p>
          </div>
          <span className="small muted mono" style={{ background: "var(--surface-2)", padding: "4px 8px", borderRadius: "var(--r-sm)" }}>
            {record.patientId} · {record.age} th · {record.sex}
          </span>
        </div>

        {/* Prediksi & Label */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 12 }}>
          <div>
            <p className="result-label" style={{ color: isGerd ? "var(--warn)" : "var(--ok)", margin: 0 }}>
              {isGerd ? "GERD" : "GERD_Normal"}
            </p>
            <p className="muted small">Berkas: {record.file}</p>
          </div>
          <div style={{ textAlign: "right" }}>
            <span className="small muted">Keyakinan (Confidence)</span>
            <div className="mono" style={{ fontSize: 24, fontWeight: 700, color: isGerd ? "var(--warn)" : "var(--ok)" }}>
              {pct(record.confidence)}
            </div>
          </div>
        </div>

        {/* Bar Keyakinan */}
        <div className="stack" style={{ ["--gap" as string]: "6px" }}>
          <div className="row" style={{ justifyContent: "space-between", fontSize: 13 }}>
            <span className="label">Tingkat Keyakinan Model</span>
            <span className="mono">{pct(record.confidence)}</span>
          </div>
          <div className={`bar ${isGerd ? "warn" : ""}`}>
            <i style={{ width: pct(record.confidence) }} />
          </div>
          <p className="hint">Ambang batas klinis Anda: {threshold}%</p>
        </div>

        {/* Probabilitas per kelas */}
        {record.probabilities && (
          <div className="stack" style={{ ["--gap" as string]: "8px" }}>
            <span className="label">Probabilitas Klasifikasi</span>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ background: "var(--surface)", padding: 10, borderRadius: "var(--r-md)" }}>
                <div className="row small" style={{ justifyContent: "space-between", marginBottom: 4 }}>
                  <span>P(GERD)</span>
                  <span className="mono" style={{ fontWeight: 600 }}>{pct(record.probabilities.GERD)}</span>
                </div>
                <div className="bar warn">
                  <i style={{ width: pct(record.probabilities.GERD) }} />
                </div>
              </div>
              <div style={{ background: "var(--surface)", padding: 10, borderRadius: "var(--r-md)" }}>
                <div className="row small" style={{ justifyContent: "space-between", marginBottom: 4 }}>
                  <span>P(GERD_Normal)</span>
                  <span className="mono" style={{ fontWeight: 600 }}>{pct(record.probabilities.GERD_Normal ?? record.probabilities.Normal)}</span>
                </div>
                <div className="bar">
                  <i style={{ width: pct(record.probabilities.GERD_Normal ?? record.probabilities.Normal) }} />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tampilan Visualisasi Grad-CAM */}
        {(record.heatmap_image || record.overlay_image || record.original_image) && (
          <div className="stack" style={{ ["--gap" as string]: "10px" }}>
            <span className="label">Visualisasi Keterbukaan Model (Grad-CAM)</span>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
              {record.original_image && (
                <div style={{ background: "#000", borderRadius: "var(--r-md)", overflow: "hidden", textAlign: "center" }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={record.original_image} alt="Citra asli" style={{ width: "100%", height: 180, objectFit: "contain", display: "block" }} />
                  <div style={{ background: "var(--surface)", padding: "4px 8px", fontSize: 11, fontWeight: 600 }}>
                    1. Citra Asli
                  </div>
                </div>
              )}
              {record.heatmap_image && (
                <div style={{ background: "#000", borderRadius: "var(--r-md)", overflow: "hidden", textAlign: "center" }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={record.heatmap_image} alt="Grad-CAM Heatmap" style={{ width: "100%", height: 180, objectFit: "contain", display: "block" }} />
                  <div style={{ background: "var(--surface)", padding: "4px 8px", fontSize: 11, fontWeight: 600 }}>
                    2. Grad-CAM Heatmap
                  </div>
                </div>
              )}
              {record.overlay_image && (
                <div style={{ background: "#000", borderRadius: "var(--r-md)", overflow: "hidden", textAlign: "center" }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={record.overlay_image} alt="Grad-CAM Overlay" style={{ width: "100%", height: 180, objectFit: "contain", display: "block" }} />
                  <div style={{ background: "var(--surface)", padding: "4px 8px", fontSize: 11, fontWeight: 600 }}>
                    3. Grad-CAM Overlay
                  </div>
                </div>
              )}
            </div>
            <p className="hint" style={{ fontSize: 12, lineHeight: 1.4 }}>
              * Grad-CAM menunjukkan area fitur citra yang paling memengaruhi keputusan model EfficientNetB4, bukan lokasi lesi patologis definitif.
            </p>
          </div>
        )}

        {low && (
          <p className="notice warn">
            Keyakinan di bawah ambang batas {threshold}%. Tinjau citra secara manual sebelum mengambil keputusan.
          </p>
        )}

        <p className="small muted">
          Batas Tanggung Jawab: Hasil keluaran AI ini merupakan alat bantu penyaringan dan bukan diagnosis medis mandiri.
        </p>
      </div>

      <div className="dlg-foot no-print">
        <button type="button" className="btn btn-ghost" onClick={() => ref.current?.close()}>
          Tutup
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => window.print()}>
          Cetak
        </button>
        {onSave && (
          <button type="button" className="btn btn-primary" onClick={onSave}>
            Simpan ke riwayat
          </button>
        )}
      </div>
    </dialog>
  );
}
