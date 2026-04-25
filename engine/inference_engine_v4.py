# ══════════════════════════════════════════════════════════════════════
# inference_engine_v4.py  ·  v4 — Severity Mapping Fix
#
# FIXES IN v4  (on top of v3)
# ──────────────────────────────
# FIX 9  Immutable Severity Mapping (CRITICAL)
#         · PHQ8_BANDS now use HALF-OPEN intervals [lo, hi) so fractional
#           scores (e.g. 4.9, 9.7, 14.3) ALWAYS resolve to the correct band.
#         · Old integer-closed [lo, hi] caused 4.9 to fall through ALL bands
#           and hit the "Severe" fallback — root cause of the bug.
#         · Severity mapping strictly locked to PHQ-8 score S:
#             0.0 <= S < 5.0   -> Minimal  (Green)
#             5.0 <= S < 10.0  -> Mild     (Yellow)
#            10.0 <= S < 15.0  -> Moderate (Orange)
#            15.0 <= S < 20.0  -> Moderately Severe (Red)
#            20.0 <= S <= 24.0 -> Severe   (Dark Red)
#
# FIX 10 Conditional Safety Override (hardened)
#         · Safety Override ONLY activates: PHQ-8 >= 10.0 AND label=HC.
#         · Explicitly disabled (False) for all scores < 10.0.
#
# NEW IN v3  (carried forward)
# ──────────────────────────────
# FIX 7  Auto Domain Intelligence
#         · YouTube URL  -> domain=1.0 (Vlog) automatically
#         · Local file   -> domain=0.0 (Clinical) by default
#         · Manual --domain flag still overrides both
#
# FIX 8  HTML Visual Dashboard
#         · Full-color browser report opens automatically on completion
#         · Color-coded PHQ-8 severity bands (green/yellow/orange/red)
#         · YouTube thumbnail fetched and displayed prominently
#         · Local file shows filename + audio format badge
#         · Temporal probability chart (inline SVG sparkline)
#         · Ensemble gauge + acoustic biomarker table
#         · Safety protocol: PHQ-8 >= 10 forces "Moderate/Severe" banner
#           ONLY when score is genuinely above threshold (fixed in v4)
#
# USAGE
# ─────
#   python inference_engine_v3.py interview.wav
#   python inference_engine_v3.py https://youtu.be/…
#   python inference_engine_v3.py interview.wav --domain 1.0
#   python inference_engine_v3.py interview.wav --no-browser   # skip auto-open
#
# All v2 programmatic API still works:
#   from inference_engine_v3 import InferencePipeline
#   pipe   = InferencePipeline("checkpoints/")
#   report = pipe.run("interview.wav")
#   html   = generate_html_report(report)
# ══════════════════════════════════════════════════════════════════════

import os, sys, json, pickle, warnings, argparse, tempfile, webbrowser
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────
DIM         = 24
MAX_SEQ_LEN = 300
DEVICE      = torch.device("cpu")

PHQ8_DEPRESSION_CUTOFF      = 10
DEFAULT_ENSEMBLE_THRESHOLD  = 0.47

TAN_TEMPERATURE = 0.6
LGB_TEMPERATURE = 0.8

EGEMAPS_FEATURE_NAMES = [
    'Loudness',        'alphaRatio',      'hammarbergIndex',
    'slope0-500',      'slope500-1500',   'spectralFlux',
    'MFCC1',           'MFCC2',           'MFCC3',           'MFCC4',
    'F0semitone',      'jitterLocal',     'shimmerLocaldB',  'HNRdBACF',
    'logRelF0-H1-H2',  'logRelF0-H1-A3',
    'F1frequency',     'F1bandwidth',     'F1amplitudeLogRelF0',
    'F2frequency',     'F2amplitudeLogRelF0',
    'F3frequency',     'F3amplitudeLogRelF0',
]

FEATURE_DESCRIPTIONS = {
    'Loudness':            'Vocal loudness / energy level',
    'alphaRatio':          'Spectral tilt (low-to-high frequency balance)',
    'hammarbergIndex':     'Hammarberg index (glottal excitation quality)',
    'slope0-500':          'Spectral slope 0–500 Hz',
    'slope500-1500':       'Spectral slope 500–1500 Hz',
    'spectralFlux':        'Spectral flux (rate of tonal change)',
    'MFCC1':               'MFCC-1 (overall spectral shape)',
    'MFCC2':               'MFCC-2 (spectral fine detail)',
    'MFCC3':               'MFCC-3 (spectral fine detail)',
    'MFCC4':               'MFCC-4 (spectral fine detail)',
    'F0semitone':          'Fundamental pitch (F0 in semitones)',
    'jitterLocal':         'Jitter (cycle-to-cycle pitch instability)',
    'shimmerLocaldB':      'Shimmer (amplitude irregularity)',
    'HNRdBACF':            'Harmonics-to-noise ratio (voice clarity)',
    'logRelF0-H1-H2':      'H1–H2 tilt (vocal breathiness)',
    'logRelF0-H1-A3':      'H1–A3 tilt (overall voice quality)',
    'F1frequency':         '1st formant frequency (vowel articulation)',
    'F1bandwidth':         '1st formant bandwidth',
    'F1amplitudeLogRelF0': '1st formant amplitude',
    'F2frequency':         '2nd formant frequency (articulation precision)',
    'F2amplitudeLogRelF0': '2nd formant amplitude',
    'F3frequency':         '3rd formant frequency',
    'F3amplitudeLogRelF0': '3rd formant amplitude',
}

DEPRESSION_DIRECTION = {
    'Loudness': -1, 'alphaRatio': -1, 'hammarbergIndex': -1,
    'slope0-500': +1, 'slope500-1500': -1, 'spectralFlux': -1,
    'MFCC1': 0, 'MFCC2': 0, 'MFCC3': 0, 'MFCC4': 0,
    'F0semitone': -1, 'jitterLocal': +1, 'shimmerLocaldB': +1,
    'HNRdBACF': -1, 'logRelF0-H1-H2': +1, 'logRelF0-H1-A3': +1,
    'F1frequency': -1, 'F1bandwidth': +1, 'F1amplitudeLogRelF0': -1,
    'F2frequency': -1, 'F2amplitudeLogRelF0': -1,
    'F3frequency': 0, 'F3amplitudeLogRelF0': -1,
}


# ═════════════════════════════════════════════════════════════════════
# FIX 7 — AUTOMATIC DOMAIN INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════

def detect_domain(source: str) -> float:
    """
    Infer the correct domain tag from the input source.

    Rules
    -----
    • YouTube URL  (http/https)  → 1.0  (D-Vlog)
    • Local file / any other     → 0.0  (Clinical / E-DAIC)

    Returns
    -------
    float : 0.0 or 1.0
    """
    if isinstance(source, str):
        if source.startswith("http://") or source.startswith("https://"):
            print("  [Auto-Domain] YouTube URL detected → domain = 1.0 (Vlog)")
            return 1.0
    print("  [Auto-Domain] Local file detected  → domain = 0.0 (Clinical)")
    return 0.0


def _extract_youtube_metadata(url: str) -> dict:
    """
    Fetch video title and thumbnail URL from a YouTube URL using yt-dlp.
    Returns dict with keys: title, thumbnail_url, channel, duration_str.
    Falls back gracefully if yt-dlp is unavailable.
    """
    try:
        import yt_dlp
        ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        duration_s = info.get('duration', 0) or 0
        m, s = divmod(int(duration_s), 60)
        h, m = divmod(m, 60)
        duration_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
        return {
            "title"         : info.get('title', 'Unknown Title'),
            "thumbnail_url" : info.get('thumbnail', ''),
            "channel"       : info.get('uploader', 'Unknown Channel'),
            "duration_str"  : duration_str,
            "url"           : url,
        }
    except Exception:
        return {
            "title"         : "YouTube Video",
            "thumbnail_url" : "",
            "channel"       : "",
            "duration_str"  : "—",
            "url"           : url,
        }


# ═════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (bit-for-bit identical to training notebook)
# ═════════════════════════════════════════════════════════════════════

class TBlock(nn.Module):
    def __init__(self, d, window, n_heads=4, dropout=0.25):
        super().__init__()
        self.w    = window
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                           batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 2, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        xw = (x[:, torch.linspace(0, T - 1, min(T, self.w))
                 .long().to(x.device), :]
              if T > self.w else x)
        xn = self.norm(xw)
        out, _ = self.attn(xn, xn, xn)
        out = xw + self.drop(out)
        return (out + self.drop(self.ff(self.norm(out)))).mean(1)


class TAN(nn.Module):
    def __init__(self, d=DIM, dm=96, n_scales=2, dropout=0.25):
        super().__init__()
        windows = [30, 60, 100, 300][:n_scales]
        self.proj   = nn.Linear(d, dm)
        self.scales = nn.ModuleList(
            [TBlock(dm, w, dropout=dropout) for w in windows])
        self.fc = nn.Sequential(
            nn.LayerNorm(dm * n_scales), nn.Dropout(dropout),
            nn.Linear(dm * n_scales, dm * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dm * 2, 1))

    def forward(self, x, mask):
        x = self.proj(x)
        return self.fc(torch.cat([sc(x) for sc in self.scales], dim=1))


# ═════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═════════════════════════════════════════════════════════════════════

class ModelLoader:
    def __init__(self, checkpoints_dir: str):
        self.ckpt_dir = checkpoints_dir
        self._verify_files()
        print("  [ModelLoader] Loading artefacts…")

        self.scaler = self._load_pkl("scaler_v3.pkl")
        print("  ✓  scaler_v3.pkl")

        arch = self._load_json("arch_configs_v3.json")["TAN"]
        self.tan_cfg = {
            "d": arch["input_dim"], "dm": arch["dm"],
            "n_scales": arch["n_scales"], "dropout": arch["dropout"],
        }

        self.tan = TAN(**self.tan_cfg).to(DEVICE)
        state = torch.load(self._path("TAN_best.pt"), map_location=DEVICE)
        self.tan.load_state_dict(state)
        self.tan.eval()
        print(f"  ✓  TAN_best.pt  (dm={self.tan_cfg['dm']}, "
              f"n_scales={self.tan_cfg['n_scales']})")

        self.lgb = self._load_pkl("lgb_v3.pkl")
        print("  ✓  lgb_v3.pkl")

        ens = self._load_json("best_ensemble_v3.json")
        self.threshold = float(ens.get("best_threshold",
                                        DEFAULT_ENSEMBLE_THRESHOLD))
        print(f"  ✓  Ensemble threshold = {self.threshold}")

    def _path(self, f): return os.path.join(self.ckpt_dir, f)
    def _load_pkl(self, f):
        with open(self._path(f), "rb") as fh: return pickle.load(fh)
    def _load_json(self, f):
        with open(self._path(f)) as fh: return json.load(fh)
    def _verify_files(self):
        missing = [r for r in ["scaler_v3.pkl","TAN_best.pt","lgb_v3.pkl",
                                "arch_configs_v3.json","best_ensemble_v3.json"]
                   if not os.path.exists(self._path(r))]
        if missing:
            raise FileNotFoundError(
                f"Missing checkpoint files: {missing}\nExpected in: {self.ckpt_dir}")


# ═════════════════════════════════════════════════════════════════════
# CALIBRATION HELPERS
# ═════════════════════════════════════════════════════════════════════

def _temperature_scale(logit: float, temperature: float) -> float:
    return float(torch.sigmoid(torch.tensor(logit / temperature)).item())

def _logit_from_prob(p: float) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.log(p / (1 - p)))

def _calibrate_lgb(prob: float) -> float:
    return _temperature_scale(_logit_from_prob(prob), LGB_TEMPERATURE)

def _confidence_weight(prob: float) -> float:
    return abs(prob - 0.5) + 1e-6


# ═════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═════════════════════════════════════════════════════════════════════

class InferenceEngine:
    def __init__(self, loader: ModelLoader):
        self.loader = loader

    def _tan_on_chunk(self, chunk, mask) -> float:
        x = torch.FloatTensor(chunk).unsqueeze(0).to(DEVICE)
        m = torch.BoolTensor(mask).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = self.loader.tan(x, m).squeeze().item()
        return _temperature_scale(logit, TAN_TEMPERATURE)

    def predict_from_features(self, enriched, dl_input, dl_mask,
                               chunks=None, chunk_masks=None,
                               chunk_starts=None) -> dict:
        feat_2d  = enriched.reshape(1, -1)
        lgb_raw  = float(self.loader.lgb.predict_proba(feat_2d)[0, 1])
        lgb_prob = _calibrate_lgb(lgb_raw)

        if chunks is not None and len(chunks) > 0:
            chunk_probs = [self._tan_on_chunk(c, m)
                           for c, m in zip(chunks, chunk_masks)]
        else:
            chunk_probs  = [self._tan_on_chunk(dl_input, dl_mask)]
            chunk_starts = [0]

        tan_prob = float(np.mean(chunk_probs))

        w_tan = _confidence_weight(tan_prob)
        w_lgb = _confidence_weight(lgb_prob)
        total_w = w_tan + w_lgb
        ensemble_prob = (w_tan * tan_prob + w_lgb * lgb_prob) / total_w

        is_depressed = ensemble_prob >= self.loader.threshold
        label        = "MDD" if is_depressed else "HC"

        return {
            "tan_prob"      : round(tan_prob,      4),
            "lgb_prob"      : round(lgb_prob,      4),
            "lgb_raw"       : round(lgb_raw,        4),
            "ensemble_prob" : round(ensemble_prob,  4),
            "label"         : label,
            "is_depressed"  : bool(is_depressed),
            "chunk_probs"   : [round(p, 4) for p in chunk_probs],
            "chunk_starts"  : chunk_starts if chunk_starts else [0],
            "tan_weight"    : round(w_tan / total_w, 3),
            "lgb_weight"    : round(w_lgb / total_w, 3),
        }


# ═════════════════════════════════════════════════════════════════════
# PHQ-8 SEVERITY BANDS
# ═════════════════════════════════════════════════════════════════════

# ── PHQ-8 Severity Bands ─────────────────────────────────────────────
# Each entry: (lo_inclusive, hi_exclusive, severity_label, info_dict)
# The lookup uses HALF-OPEN intervals: lo <= score < hi
# so that fractional scores (e.g. 4.9, 9.7) always resolve correctly.
# hi for the last band is 25.0 to capture the maximum value of 24.0.
PHQ8_BANDS = [
    (0.0, 5.0, "Minimal", {
        "headline": "No significant depressive symptoms detected.",
        "what_it_means": (
            "The acoustic profile is consistent with a healthy emotional baseline. "
            "Vocal patterns show normal variation in pitch, energy, and articulation "
            "with no markers strongly associated with clinical depression."
        ),
        "immediate_action": "No clinical intervention is required at this time.",
        "self_care": [
            "Maintain regular physical activity (≥150 min/week moderate exercise).",
            "Prioritise consistent sleep hygiene (7–9 hours per night).",
            "Nurture social connections and meaningful activities.",
            "Continue any existing mental wellness practices.",
        ],
        "follow_up": "Re-screen if mood or sleep changes significantly over 2+ weeks.",
        "urgency_color": "#22c55e",
        "urgency_emoji": "🟢",
        "css_class": "minimal",
    }),
    (5.0, 10.0, "Mild", {
        "headline": "Mild depressive signals detected.",
        "what_it_means": (
            "The vocal analysis has picked up subtle but consistent changes in pitch "
            "dynamics, vocal energy, and spectral characteristics that can accompany "
            "low mood, fatigue, or early-stage emotional distress."
        ),
        "immediate_action": (
            "Consider speaking with a trusted person (friend, family, GP) "
            "about how you have been feeling lately."
        ),
        "self_care": [
            "Try structured mood-tracking (e.g. journalling) for 2 weeks.",
            "Reduce alcohol and caffeine intake.",
            "Engage in at least one pleasant or social activity per day.",
            "Practise brief mindfulness or breathing exercises (10 min/day).",
        ],
        "follow_up": "Re-screen in 4 weeks. If symptoms persist, consult a GP.",
        "urgency_color": "#eab308",
        "urgency_emoji": "🟡",
        "css_class": "mild",
    }),
    (10.0, 15.0, "Moderate", {
        "headline": "Moderate depressive signals detected.",
        "what_it_means": (
            "Acoustic biomarkers — including reduced vocal energy, lowered pitch "
            "variability, and altered voice quality — are consistent with patterns seen "
            "in individuals meeting criteria for a moderate depressive episode."
        ),
        "immediate_action": (
            "Consultation with a General Practitioner or primary care provider "
            "is recommended within the next 1–2 weeks."
        ),
        "self_care": [
            "Do not wait to reach out — speak to a doctor soon.",
            "Structured psychological support (CBT, counselling) is beneficial.",
            "Maintain a routine, even if motivation feels very low.",
            "Limit isolation; schedule regular contact with supportive people.",
        ],
        "follow_up": "Follow up with a clinician within 2 weeks.",
        "urgency_color": "#f97316",
        "urgency_emoji": "🟠",
        "css_class": "moderate",
    }),
    (15.0, 20.0, "Moderately Severe", {
        "headline": "Moderately severe depressive signals detected.",
        "what_it_means": (
            "Strong acoustic markers of depression are present across multiple dimensions: "
            "significantly reduced loudness and pitch range, elevated vocal irregularity, "
            "and reduced spectral dynamics."
        ),
        "immediate_action": (
            "Referral to a mental health professional (psychologist or psychiatrist) "
            "is strongly recommended. Do not delay."
        ),
        "self_care": [
            "Reach out to a mental health crisis line if you feel unsafe.",
            "Inform a trusted person about how you are feeling.",
            "Avoid making irreversible decisions during this period.",
            "Structured psychotherapy (CBT, IPT) or pharmacotherapy is indicated.",
        ],
        "follow_up": "Schedule a mental health appointment within 1 week.",
        "urgency_color": "#ef4444",
        "urgency_emoji": "🔴",
        "css_class": "mod-severe",
    }),
    (20.0, 25.0, "Severe", {
        "headline": "Severe depressive signals detected — urgent attention required.",
        "what_it_means": (
            "The acoustic profile shows extreme deviation from healthy speech patterns "
            "across nearly all measured dimensions. This level is associated with severe "
            "functional impairment and may indicate an acute depressive episode."
        ),
        "immediate_action": (
            "⚠️ URGENT: Seek immediate evaluation from a psychiatrist or mental health "
            "crisis team. If you are in crisis, call a crisis helpline or go to your "
            "nearest emergency department."
        ),
        "self_care": [
            "Do not be alone — contact someone you trust right now.",
            "Call a crisis line: iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345",
            "International resources: https://www.iasp.info/resources/Crisis_Centres/",
            "Suicide risk assessment by a clinician is essential.",
        ],
        "follow_up": "Do not wait. Immediate psychiatric evaluation is required.",
        "urgency_color": "#7f1d1d",
        "urgency_emoji": "🚨",
        "css_class": "severe",
    }),
]

def _get_phq8_band(phq8_est: float) -> dict:
    """
    Map a PHQ-8 estimate to its severity band.

    Uses HALF-OPEN intervals [lo, hi) so fractional scores such as 4.9
    always resolve to the correct band instead of falling through to the
    fallback ('Severe').  The final band's hi is 25.0, which safely
    captures the maximum possible estimate of 24.0.
    """
    for lo, hi, severity, info in PHQ8_BANDS:
        if lo <= phq8_est < hi:          # ← half-open: lo inclusive, hi exclusive
            return {"severity": severity, "lo": lo, "hi": hi, **info}
    # Unreachable in practice (score is clamped to [0.5, 23.5] upstream),
    # but kept as a safe fallback.
    lo, hi, severity, info = PHQ8_BANDS[-1]
    return {"severity": severity, "lo": lo, "hi": hi, **info}


# ═════════════════════════════════════════════════════════════════════
# AGENTIC REASONING LAYER
# ═════════════════════════════════════════════════════════════════════

class AgentLayer:
    def __init__(self, scaler):
        self.scaler = scaler

    def generate_report(self, prediction, feat_norm, duration_sec,
                         segment_feat_norms=None, chunk_starts=None,
                         fps=100.0) -> dict:

        prob   = prediction["ensemble_prob"]
        is_dep = prediction["is_depressed"]

        phq8_est = round(max(0.5, min(23.5, prob * 24.0)), 1)

        # ── Safety Override (Immutable Rule) ──────────────────────────
        # ONLY activates when BOTH conditions are true simultaneously:
        #   1. PHQ-8 estimate >= 10.0  (at-or-above the clinical cutoff)
        #   2. The ensemble labelled the case HC  (classification mismatch)
        # If PHQ-8 < 10.0, this block is COMPLETELY DISABLED regardless of
        # any other condition.  This prevents the "Severe" banner from ever
        # appearing for scores below the clinical threshold.
        if (phq8_est >= float(PHQ8_DEPRESSION_CUTOFF)) and (not is_dep):
            safety_override = True
        else:
            safety_override = False   # ← explicitly False for PHQ-8 < 10.0

        band = _get_phq8_band(phq8_est)

        session_mean = feat_norm.mean(axis=0)
        z_scores     = session_mean
        global_signals = self._get_acoustic_signals(z_scores, is_dep, top_n=5)

        temporal_info = self._temporal_analysis(
            prediction.get("chunk_probs", [prob]),
            prediction.get("chunk_starts", [0]),
            segment_feat_norms or [feat_norm],
            fps, duration_sec, is_dep
        )

        confidence_level, confidence_pct = self._compute_confidence(prob)

        report = {
            "label"              : prediction["label"],
            "is_depressed"       : is_dep,
            "safety_override"    : safety_override,
            "ensemble_prob"      : prob,
            "tan_prob"           : prediction["tan_prob"],
            "lgb_prob"           : prediction["lgb_prob"],
            "tan_weight"         : prediction.get("tan_weight", 0.5),
            "lgb_weight"         : prediction.get("lgb_weight", 0.5),
            "n_chunks_analysed"  : len(prediction.get("chunk_probs", [1])),
            "chunk_probs"        : prediction.get("chunk_probs", [prob]),
            "phq8_estimate"      : phq8_est,
            "phq8_cutoff"        : PHQ8_DEPRESSION_CUTOFF,
            "severity"           : band["severity"],
            "band_info"          : band,
            "confidence_level"   : confidence_level,
            "confidence_pct"     : confidence_pct,
            "acoustic_signals"   : global_signals,
            "temporal_info"      : temporal_info,
            "duration_sec"       : round(duration_sec, 1),
        }

        self._print_report(report)
        return report

    def _get_acoustic_signals(self, z_scores, is_dep, top_n=5):
        signals = []
        for i, name in enumerate(EGEMAPS_FEATURE_NAMES):
            z    = float(z_scores[i])
            dirn = DEPRESSION_DIRECTION.get(name, 0)
            desc = FEATURE_DESCRIPTIONS.get(name, name)
            if dirn == 0:
                consistent = True
            elif is_dep:
                consistent = (dirn > 0 and z > 0) or (dirn < 0 and z < 0)
            else:
                consistent = (dirn > 0 and z < 0) or (dirn < 0 and z > 0)
            signals.append({
                "feature"    : name,
                "description": desc,
                "z_score"    : round(z, 3),
                "consistent" : consistent,
            })
        signals.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return signals[:top_n]

    def _temporal_analysis(self, chunk_probs, chunk_starts,
                            segment_feat_norms, fps,
                            duration_sec, is_dep) -> dict:
        if not chunk_probs:
            return {"summary": "Insufficient audio for temporal analysis."}

        n     = len(chunk_probs)
        probs = np.array(chunk_probs)

        peak_idx   = int(np.argmax(probs))
        trough_idx = int(np.argmin(probs))

        def _frame_to_time(frame_idx):
            secs = frame_idx / fps
            mins = int(secs // 60)
            secs_rem = secs % 60
            if mins > 0:
                return f"{mins}m {secs_rem:.0f}s"
            return f"{secs_rem:.1f}s"

        peak_start_f = chunk_starts[peak_idx] if chunk_starts else 0
        peak_time    = _frame_to_time(peak_start_f)
        peak_end_t   = _frame_to_time(peak_start_f + 300)

        if segment_feat_norms and peak_idx < len(segment_feat_norms):
            seg_feat = segment_feat_norms[peak_idx]
            if seg_feat.shape[0] > 0:
                seg_z = seg_feat.mean(axis=0)
                peak_signals = self._get_acoustic_signals(seg_z, is_dep, top_n=3)
            else:
                peak_signals = []
        else:
            peak_signals = []

        variance = float(np.var(probs))
        trend    = ("increasing" if probs[-1] > probs[0] + 0.05 else
                    "decreasing" if probs[0] > probs[-1] + 0.05 else "stable")

        return {
            "n_chunks"          : n,
            "chunk_probs"       : [round(float(p), 4) for p in probs],
            "peak_prob"         : round(float(probs[peak_idx]), 4),
            "peak_window_start" : peak_time,
            "peak_window_end"   : peak_end_t,
            "peak_signals"      : peak_signals,
            "trough_prob"       : round(float(probs[trough_idx]), 4),
            "variance"          : round(variance, 4),
            "trend"             : trend,
            "mean_prob"         : round(float(probs.mean()), 4),
        }

    def _compute_confidence(self, prob):
        dist = abs(prob - DEFAULT_ENSEMBLE_THRESHOLD)
        pct  = round(min(100.0, dist / 0.53 * 100), 1)
        if dist < 0.05:
            return "Low — Borderline", pct
        elif dist < 0.15:
            return "Moderate", pct
        else:
            return "High", pct

    def _print_report(self, r):
        W   = 70
        DIV = "─" * W
        EQ  = "═" * W
        band = r["band_info"]

        def line(txt="", pad=2):
            prefix = " " * pad
            while len(txt) > W - pad - 2:
                cut = txt[:W - pad - 2].rfind(" ")
                if cut == -1: cut = W - pad - 2
                print(f"{prefix}{txt[:cut]}")
                txt = txt[cut:].lstrip()
            print(f"{prefix}{txt}")

        def section(title):
            print(f"\n{DIV}")
            print(f"  {title}")
            print(DIV)

        print("\n" + EQ)
        print("  🧠  ACOUSTIC DEPRESSION SCREENING REPORT  v4")
        print("      Research Tool — Not a Clinical Diagnosis")
        print(EQ)

        phq = r["phq8_estimate"]
        label_color = "🔴" if r["is_depressed"] else "🟢"
        print()
        print(f"  {'PHQ-8 Estimate':<22}: {phq:>5.1f} / 24   (cutoff ≥ {r['phq8_cutoff']})")
        print(f"  {'Severity Band':<22}: {band['urgency_emoji']}  {r['severity']}")
        if r.get("safety_override"):
            print(f"\n  ⚠️  SAFETY PROTOCOL ACTIVE: PHQ-8 ≥ 10 — Moderate/Severe flag applied")
        print()
        print(f"  {'─'*66}")
        print(f"  {label_color}  CLASSIFICATION :  {r['label']}")
        print(f"  {'─'*66}")

        section("📊  MODEL SCORES & ENSEMBLE")
        print(f"  {'Deep Learning (TAN)':<28}: {r['tan_prob']:.4f}  (weight = {r['tan_weight']:.2f})")
        print(f"  {'Classical ML (LightGBM)':<28}: {r['lgb_prob']:.4f}  (weight = {r['lgb_weight']:.2f})")
        print(f"  {'Ensemble (weighted)':<28}: {r['ensemble_prob']:.4f}")
        print(f"  {'Decision Threshold':<28}: {DEFAULT_ENSEMBLE_THRESHOLD}")
        print(f"  {'Confidence':<28}: {r['confidence_level']}  ({r['confidence_pct']}%)")
        print(f"  {'Chunks Analysed (TAN)':<28}: {r['n_chunks_analysed']} × 3-second windows")
        print(f"  {'Audio Duration':<28}: {r['duration_sec']}s")

        section("💡  WHAT THIS MEANS")
        line(band["what_it_means"])

        section("🎙️   TOP ACOUSTIC BIOMARKERS")
        for s in r["acoustic_signals"]:
            arrow = "↑" if s["z_score"] > 0 else "↓"
            tick  = "✓" if s["consistent"] else "↔"
            print(f"  {tick} {arrow} {s['description']:<45} z = {s['z_score']:+.2f}")

        t = r["temporal_info"]
        if t.get("n_chunks", 0) > 1:
            section("⏱️   TEMPORAL ANALYSIS")
            print(f"  Strongest signal  : {t['peak_window_start']} – {t['peak_window_end']}")
            print(f"  Peak probability  : {t['peak_prob']:.4f}")
            print(f"  Signal trend      : {t['trend'].capitalize()}")

        section(f"📋  CLINICAL RECOMMENDATIONS  [{r['severity']}]")
        line(band["headline"])
        print(f"\n  Immediate Action:")
        line(band["immediate_action"], pad=4)
        print(f"\n  Self-Care & Lifestyle:")
        for item in band["self_care"]:
            line(f"• {item}", pad=4)
        print(f"\n  Follow-Up:")
        line(band["follow_up"], pad=4)

        print()
        print(EQ)
        print("  ⚠️   IMPORTANT DISCLAIMER")
        print(DIV)
        line("This report is generated by a research-grade acoustic screening "
             "model. It is NOT a clinical diagnosis and must NOT replace "
             "professional mental health evaluation.")
        print(EQ + "\n")
        print("  🌐  Opening visual dashboard in browser…")


# ═════════════════════════════════════════════════════════════════════
# FIX 8 — HTML VISUAL DASHBOARD GENERATOR
# ═════════════════════════════════════════════════════════════════════

def _build_sparkline_svg(chunk_probs: list, width=480, height=80) -> str:
    """Build an inline SVG sparkline of per-chunk probabilities."""
    if not chunk_probs or len(chunk_probs) < 2:
        return ""
    n = len(chunk_probs)
    pad = 10
    w_inner = width - 2 * pad
    h_inner = height - 2 * pad

    def _x(i):
        return pad + i * w_inner / (n - 1)

    def _y(p):
        return pad + (1.0 - p) * h_inner

    # Gradient area path
    area_pts = [f"M {_x(0):.1f},{height}"]
    for i, p in enumerate(chunk_probs):
        area_pts.append(f"L {_x(i):.1f},{_y(p):.1f}")
    area_pts.append(f"L {_x(n-1):.1f},{height} Z")

    # Line path
    line_pts = [f"M {_x(0):.1f},{_y(chunk_probs[0]):.1f}"]
    for i, p in enumerate(chunk_probs[1:], 1):
        line_pts.append(f"L {_x(i):.1f},{_y(p):.1f}")

    # Threshold line at 0.47
    thresh_y = _y(DEFAULT_ENSEMBLE_THRESHOLD)

    svg = f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:{height}px">
  <defs>
    <linearGradient id="sparkGrad" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="var(--accent)" stop-opacity="0.02"/>
    </linearGradient>
  </defs>
  <path d="{' '.join(area_pts)}" fill="url(#sparkGrad)"/>
  <line x1="{pad}" y1="{thresh_y:.1f}" x2="{width-pad}" y2="{thresh_y:.1f}"
        stroke="#6b7280" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>
  <path d="{' '.join(line_pts)}" fill="none" stroke="var(--accent)" stroke-width="2.2" stroke-linejoin="round"/>
  {''.join(f'<circle cx="{_x(i):.1f}" cy="{_y(p):.1f}" r="3" fill="var(--accent)"/>' for i, p in enumerate(chunk_probs))}
</svg>"""
    return svg


def _gauge_svg(prob: float, threshold: float = DEFAULT_ENSEMBLE_THRESHOLD,
               color: str = "#ef4444") -> str:
    """Build a semicircular probability gauge."""
    import math
    cx, cy, r = 100, 100, 72
    start_angle = 180
    sweep = 180 * prob
    end_angle = start_angle + sweep
    rad = math.radians
    x1 = cx + r * math.cos(rad(start_angle))
    y1 = cy + r * math.sin(rad(start_angle))
    x2 = cx + r * math.cos(rad(end_angle))
    y2 = cy + r * math.sin(rad(end_angle))
    large = 1 if sweep > 180 else 0
    # Threshold marker
    tx = cx + r * math.cos(rad(180 + 180 * threshold))
    ty = cy + r * math.sin(rad(180 + 180 * threshold))
    tx2 = cx + (r + 12) * math.cos(rad(180 + 180 * threshold))
    ty2 = cy + (r + 12) * math.sin(rad(180 + 180 * threshold))

    return f"""<svg viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg" style="width:200px;height:110px">
  <path d="M {x1:.1f},{y1:.1f} A {r},{r} 0 1 1 {cx+r},{cy}" fill="none" stroke="#1f2937" stroke-width="10" stroke-linecap="round"/>
  <path d="M {cx-r},{cy} A {r},{r} 0 {large} 1 {x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"/>
  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="#6b7280" stroke-width="2"/>
  <text x="{cx}" y="{cy-8}" text-anchor="middle" font-family="monospace" font-size="22" font-weight="700" fill="{color}">{prob:.3f}</text>
  <text x="{cx}" y="{cy+10}" text-anchor="middle" font-family="monospace" font-size="10" fill="#9ca3af">ensemble prob</text>
</svg>"""


def generate_html_report(report: dict,
                          source_meta: dict = None) -> str:
    """
    Generate a complete, self-contained HTML dashboard from a report dict.

    Parameters
    ----------
    report      : dict returned by AgentLayer.generate_report()
    source_meta : dict with source information. Keys:
                    source_type  : "youtube" | "file"
                    filename     : str  (for local files)
                    file_format  : str  (e.g. "WAV", "MP3")
                    yt_title     : str
                    yt_channel   : str
                    yt_thumbnail : str  (URL)
                    yt_url       : str
                    yt_duration  : str
                    domain_tag   : float
    Returns
    -------
    str : full HTML document
    """
    if source_meta is None:
        source_meta = {"source_type": "file", "filename": "Unknown", "file_format": "—"}

    band      = report["band_info"]
    phq       = report["phq8_estimate"]
    severity  = report["severity"]
    color     = band["urgency_color"]
    css_class = band.get("css_class", "minimal")
    is_dep    = report["is_depressed"]
    safety_ov = report.get("safety_override", False)
    t         = report["temporal_info"]

    # Safety override — force severity display even if label is HC
    display_label = report["label"]
    if safety_ov:
        display_label = "⚠️ MDD (Safety Override)"

    sparkline = _build_sparkline_svg(report.get("chunk_probs", []))
    gauge     = _gauge_svg(report["ensemble_prob"], color=color)

    # ── Source header block ──────────────────────────────────────────
    if source_meta.get("source_type") == "youtube":
        thumb = source_meta.get("yt_thumbnail", "")
        thumb_html = (f'<img src="{thumb}" alt="Video thumbnail" class="yt-thumb"/>'
                      if thumb else '<div class="yt-thumb-placeholder">🎬</div>')
        source_html = f"""
        <div class="source-card youtube-card">
          <div class="source-left">
            {thumb_html}
          </div>
          <div class="source-right">
            <div class="source-badge">YouTube</div>
            <h2 class="source-title">{source_meta.get('yt_title','YouTube Video')}</h2>
            <div class="source-sub">{source_meta.get('yt_channel','')}</div>
            <div class="source-meta-row">
              <span>⏱ {source_meta.get('yt_duration','—')}</span>
              <span>🔗 <a href="{source_meta.get('yt_url','#')}" target="_blank">Open Video</a></span>
              <span class="domain-badge vlog">Domain: Vlog (1.0)</span>
            </div>
          </div>
        </div>"""
    else:
        fname = source_meta.get("filename", "Unknown File")
        fmt   = source_meta.get("file_format", "").upper() or "AUDIO"
        dom   = source_meta.get("domain_tag", 0.0)
        domain_label = "Vlog (1.0)" if dom == 1.0 else "Clinical (0.0)"
        source_html = f"""
        <div class="source-card file-card">
          <div class="file-icon">🎙️</div>
          <div class="source-right">
            <div class="source-badge">Local File</div>
            <h2 class="source-title">{fname}</h2>
            <div class="source-meta-row">
              <span class="format-badge">.{fmt}</span>
              <span>⏱ {report['duration_sec']}s analyzed</span>
              <span class="domain-badge {'vlog' if dom==1.0 else 'clinical'}">Domain: {domain_label}</span>
            </div>
          </div>
        </div>"""

    # ── Safety banner ────────────────────────────────────────────────
    safety_banner = ""
    if safety_ov:
        safety_banner = f"""
        <div class="safety-banner">
          <span class="safety-icon">⚠️</span>
          <div>
            <strong>Safety Protocol Active</strong><br>
            PHQ-8 Estimate ({phq:.1f}) exceeds the clinical cutoff of {PHQ8_DEPRESSION_CUTOFF}.
            Moderate-to-Severe status is flagged regardless of the borderline classification label.
          </div>
        </div>"""

    # ── Acoustic signals table ───────────────────────────────────────
    sig_rows = ""
    for s in report.get("acoustic_signals", []):
        arrow   = "↑" if s["z_score"] > 0 else "↓"
        tick    = "✓" if s["consistent"] else "↔"
        z_class = "z-pos" if s["z_score"] > 0 else "z-neg"
        sig_rows += f"""
          <tr>
            <td class="sig-tick {'consistent' if s['consistent'] else 'neutral'}">{tick}</td>
            <td class="sig-name">{s['description']}</td>
            <td class="{z_class}">{arrow} {abs(s['z_score']):.3f}</td>
          </tr>"""

    # ── Temporal section ─────────────────────────────────────────────
    temporal_html = ""
    if t.get("n_chunks", 0) > 1:
        trend_icon = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(
            t.get("trend","stable"), "➡️")
        temporal_html = f"""
        <section class="section">
          <h3 class="section-title">⏱ Temporal Reasoning — Peak Symptom Window</h3>
          <div class="temporal-grid">
            <div class="t-card">
              <div class="t-label">Peak Window</div>
              <div class="t-value">{t.get('peak_window_start','—')} – {t.get('peak_window_end','—')}</div>
            </div>
            <div class="t-card">
              <div class="t-label">Peak Probability</div>
              <div class="t-value">{t.get('peak_prob',0):.4f}</div>
            </div>
            <div class="t-card">
              <div class="t-label">Signal Trend</div>
              <div class="t-value">{trend_icon} {t.get('trend','—').capitalize()}</div>
            </div>
            <div class="t-card">
              <div class="t-label">Windows Analyzed</div>
              <div class="t-value">{t.get('n_chunks',1)}</div>
            </div>
          </div>
          <div class="spark-wrap">
            <div class="spark-label">Probability per 3-second window <span class="spark-legend">— threshold ({DEFAULT_ENSEMBLE_THRESHOLD})</span></div>
            {sparkline}
          </div>
        </section>"""

    # ── Self-care list ───────────────────────────────────────────────
    sc_items = "".join(f"<li>{item}</li>" for item in band["self_care"])

    # ── Recommendations section ──────────────────────────────────────
    recs_html = f"""
        <section class="section recs-section {css_class}-recs">
          <h3 class="section-title">📋 Clinical Tiered Advice — {severity}</h3>
          <div class="rec-block">
            <div class="rec-label">Immediate Action</div>
            <div class="rec-text">{band['immediate_action']}</div>
          </div>
          <div class="rec-block">
            <div class="rec-label">Self-Care & Lifestyle</div>
            <ul class="rec-list">{sc_items}</ul>
          </div>
          <div class="rec-block">
            <div class="rec-label">Follow-Up</div>
            <div class="rec-text">{band['follow_up']}</div>
          </div>
        </section>"""

    # ── PHQ-8 progress bar ───────────────────────────────────────────
    phq_pct = phq / 24.0 * 100

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Acoustic Depression Screening Report v4 — {severity}</title>
<style>
  :root {{
    --bg:      #0a0e1a;
    --surface: #111827;
    --border:  #1f2937;
    --text:    #f1f5f9;
    --muted:   #6b7280;
    --accent:  {color};
    --font-mono: 'Courier New', 'Lucida Console', monospace;
    --font-sans: 'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    min-height: 100vh;
    padding: 0 0 60px;
  }}

  /* ── Top bar ────────────────────────────────────── */
  .top-bar {{
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border-bottom: 1px solid var(--border);
    padding: 18px 40px;
    display: flex;
    align-items: center;
    gap: 14px;
  }}
  .top-bar-icon {{ font-size: 28px; }}
  .top-bar-title {{
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--muted);
    letter-spacing: .08em;
    text-transform: uppercase;
  }}
  .top-bar-title strong {{
    display: block;
    font-size: 18px;
    color: var(--text);
    letter-spacing: .03em;
  }}
  .top-bar-disclaimer {{
    margin-left: auto;
    font-size: 11px;
    color: var(--muted);
    text-align: right;
    max-width: 220px;
    line-height: 1.5;
  }}

  /* ── Layout ─────────────────────────────────────── */
  .container {{ max-width: 900px; margin: 0 auto; padding: 0 24px; }}

  /* ── Source card ─────────────────────────────────── */
  .source-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 28px 0 0;
    display: flex;
    gap: 20px;
    align-items: flex-start;
  }}
  .yt-thumb {{
    width: 180px;
    height: 102px;
    object-fit: cover;
    border-radius: 8px;
    flex-shrink: 0;
    border: 1px solid var(--border);
  }}
  .yt-thumb-placeholder {{
    width: 180px;
    height: 102px;
    border-radius: 8px;
    background: var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 36px;
    flex-shrink: 0;
  }}
  .file-icon {{
    font-size: 52px;
    flex-shrink: 0;
    line-height: 1;
    padding-top: 4px;
  }}
  .source-right {{ flex: 1; min-width: 0; }}
  .source-badge {{
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 2px 8px;
    margin-bottom: 6px;
  }}
  .source-title {{
    font-size: 17px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .source-sub {{ font-size: 13px; color: var(--muted); margin-bottom: 10px; }}
  .source-meta-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--muted);
    align-items: center;
  }}
  .source-meta-row a {{ color: var(--accent); text-decoration: none; }}
  .format-badge, .domain-badge {{
    background: var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 600;
    font-family: var(--font-mono);
    font-size: 11px;
  }}
  .domain-badge.vlog {{ color: #a78bfa; }}
  .domain-badge.clinical {{ color: #67e8f9; }}

  /* ── Safety banner ───────────────────────────────── */
  .safety-banner {{
    background: rgba(239,68,68,.12);
    border: 1.5px solid #ef4444;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 20px 0 0;
    display: flex;
    gap: 14px;
    align-items: flex-start;
    font-size: 14px;
    line-height: 1.6;
  }}
  .safety-icon {{ font-size: 22px; flex-shrink: 0; }}

  /* ── Primary result hero ─────────────────────────── */
  .hero {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 12px;
    margin: 20px 0 0;
    padding: 28px 28px 24px;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 20px;
    align-items: center;
  }}
  .hero-left {{}}
  .phq-number {{
    font-family: var(--font-mono);
    font-size: 64px;
    font-weight: 900;
    color: var(--accent);
    line-height: 1;
    letter-spacing: -.02em;
  }}
  .phq-label {{
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--muted);
    margin-top: 4px;
    letter-spacing: .06em;
  }}
  .severity-chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: color-mix(in srgb, var(--accent) 15%, transparent);
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 6px 14px;
    font-family: var(--font-mono);
    font-size: 14px;
    font-weight: 700;
    color: var(--accent);
    margin-top: 14px;
    letter-spacing: .05em;
    text-transform: uppercase;
  }}
  .classification-chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--border);
    border-radius: 6px;
    padding: 6px 14px;
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--muted);
    margin-top: 8px;
  }}
  .classification-chip strong {{ color: var(--text); }}
  .phq-bar-wrap {{ margin-top: 18px; }}
  .phq-bar-bg {{
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
  }}
  .phq-bar-fill {{
    height: 100%;
    width: {phq_pct:.1f}%;
    background: linear-gradient(90deg, #22c55e 0%, #eab308 30%, #f97316 60%, #ef4444 80%, #7f1d1d 100%);
    border-radius: 4px;
    transition: width .8s ease;
  }}
  .phq-bar-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--muted);
    font-family: var(--font-mono);
    margin-top: 4px;
  }}
  .hero-right {{ text-align: center; }}

  /* ── Section ──────────────────────────────────────── */
  .section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px;
    margin-top: 16px;
  }}
  .section-title {{
    font-family: var(--font-mono);
    font-size: 12px;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }}

  /* ── Ensemble grid ───────────────────────────────── */
  .ensemble-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  @media (max-width: 600px) {{ .ensemble-grid {{ grid-template-columns: 1fr 1fr; }} }}
  .e-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
  }}
  .e-label {{ font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
  .e-value {{
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
  }}
  .e-sub {{ font-size: 10px; color: var(--muted); margin-top: 2px; }}

  /* ── Acoustic signals table ───────────────────────── */
  .sig-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .sig-table td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); }}
  .sig-table tr:last-child td {{ border-bottom: none; }}
  .sig-tick {{ font-size: 16px; width: 28px; }}
  .sig-tick.consistent {{ color: var(--accent); }}
  .sig-tick.neutral {{ color: var(--muted); }}
  .sig-name {{ color: var(--text); }}
  .z-pos {{ color: #ef4444; font-family: var(--font-mono); font-weight: 600; text-align: right; }}
  .z-neg {{ color: #60a5fa; font-family: var(--font-mono); font-weight: 600; text-align: right; }}

  /* ── Temporal ─────────────────────────────────────── */
  .temporal-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 18px;
  }}
  @media (max-width: 600px) {{ .temporal-grid {{ grid-template-columns: 1fr 1fr; }} }}
  .t-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
  }}
  .t-label {{ font-size: 10px; color: var(--muted); margin-bottom: 4px; letter-spacing: .06em; text-transform: uppercase; }}
  .t-value {{ font-family: var(--font-mono); font-size: 15px; font-weight: 700; color: var(--accent); }}
  .spark-wrap {{ margin-top: 8px; }}
  .spark-label {{ font-size: 11px; color: var(--muted); margin-bottom: 6px; }}
  .spark-legend {{ color: #4b5563; }}

  /* ── Recommendations ─────────────────────────────── */
  .recs-section {{ border-top: 3px solid var(--accent); }}
  .rec-block {{ margin-bottom: 18px; }}
  .rec-block:last-child {{ margin-bottom: 0; }}
  .rec-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
  }}
  .rec-text {{ font-size: 14px; color: var(--text); line-height: 1.65; }}
  .rec-list {{ padding-left: 18px; }}
  .rec-list li {{ font-size: 14px; color: var(--text); line-height: 1.7; margin-bottom: 2px; }}

  /* ── Disclaimer ────────────────────────────────────── */
  .disclaimer {{
    background: rgba(17,24,39,.7);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 16px;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
    text-align: center;
  }}
  .disclaimer strong {{ color: #f59e0b; }}

  /* ── What it means card ───────────────────────────── */
  .what-card {{
    background: color-mix(in srgb, var(--accent) 8%, var(--surface));
    border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
    border-radius: 10px;
    padding: 16px 18px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text);
  }}
</style>
</head>
<body>

<div class="top-bar">
  <div class="top-bar-icon">🧠</div>
  <div class="top-bar-title">
    <strong>Acoustic Depression Screening Report</strong>
    Research Tool · v4 · {__import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")}
  </div>
  <div class="top-bar-disclaimer">
    Not a clinical diagnosis.<br>For research use only.
  </div>
</div>

<div class="container">

  {source_html}

  {safety_banner}

  <!-- Hero / Primary Results -->
  <div class="hero">
    <div class="hero-left">
      <div class="phq-number">{phq:.1f}</div>
      <div class="phq-label">PHQ-8 ESTIMATE  /  24   (cutoff ≥ {PHQ8_DEPRESSION_CUTOFF})</div>
      <div class="severity-chip">{band['urgency_emoji']}  {severity}</div>
      <div class="classification-chip">Classification: <strong>&nbsp;{display_label}</strong></div>
      <div class="phq-bar-wrap">
        <div class="phq-bar-bg"><div class="phq-bar-fill"></div></div>
        <div class="phq-bar-labels">
          <span>0 Minimal</span><span>5 Mild</span><span>10 Moderate</span><span>15 M.Severe</span><span>24 Severe</span>
        </div>
      </div>
    </div>
    <div class="hero-right">
      {gauge}
      <div style="font-size:11px;color:var(--muted);margin-top:4px;font-family:var(--font-mono)">Confidence: {report['confidence_level']}<br>({report['confidence_pct']}%)</div>
    </div>
  </div>

  <!-- What It Means -->
  <section class="section">
    <h3 class="section-title">💡 What This Means</h3>
    <div class="what-card">{band['what_it_means']}</div>
  </section>

  <!-- Ensemble breakdown -->
  <section class="section">
    <h3 class="section-title">📊 Ensemble Breakdown</h3>
    <div class="ensemble-grid">
      <div class="e-card">
        <div class="e-label">TAN (Deep Learning)</div>
        <div class="e-value">{report['tan_prob']:.4f}</div>
        <div class="e-sub">weight {report['tan_weight']:.2f}</div>
      </div>
      <div class="e-card">
        <div class="e-label">LightGBM (Classical)</div>
        <div class="e-value">{report['lgb_prob']:.4f}</div>
        <div class="e-sub">weight {report['lgb_weight']:.2f}</div>
      </div>
      <div class="e-card">
        <div class="e-label">Ensemble Final</div>
        <div class="e-value" style="color:var(--accent)">{report['ensemble_prob']:.4f}</div>
        <div class="e-sub">threshold {DEFAULT_ENSEMBLE_THRESHOLD}</div>
      </div>
      <div class="e-card">
        <div class="e-label">Windows Analyzed</div>
        <div class="e-value">{report['n_chunks_analysed']}</div>
        <div class="e-sub">× 3-s chunks (TAN)</div>
      </div>
      <div class="e-card">
        <div class="e-label">Audio Duration</div>
        <div class="e-value">{report['duration_sec']}s</div>
        <div class="e-sub">total analyzed</div>
      </div>
      <div class="e-card">
        <div class="e-label">Domain Tag</div>
        <div class="e-value" style="font-size:16px">{'Vlog' if source_meta.get('domain_tag',0)==1.0 else 'Clinical'}</div>
        <div class="e-sub">Feature #24</div>
      </div>
    </div>
  </section>

  <!-- Acoustic biomarkers -->
  <section class="section">
    <h3 class="section-title">🎙 Top Acoustic Biomarkers</h3>
    <table class="sig-table">
      <tbody>{sig_rows}</tbody>
    </table>
  </section>

  {temporal_html}

  {recs_html}

  <div class="disclaimer">
    <strong>⚠️ Important Disclaimer</strong><br>
    This report is generated by a research-grade acoustic screening model trained on E-DAIC and D-Vlog datasets.
    It is <strong>NOT a clinical diagnosis</strong> and must <strong>NOT</strong> be used as a substitute for
    professional mental health evaluation. Always consult a qualified clinician.
  </div>

</div>
</body>
</html>"""
    return html


def save_and_open_report(report: dict,
                          source_meta: dict = None,
                          output_path: str = None,
                          auto_open: bool = True) -> str:
    """
    Render the HTML report, save it to a temp file, and open in browser.

    Parameters
    ----------
    report      : dict from AgentLayer.generate_report()
    source_meta : dict with source metadata
    output_path : optional explicit path; if None, a temp file is used
    auto_open   : if True, open the browser automatically

    Returns
    -------
    str : path to the saved HTML file
    """
    html = generate_html_report(report, source_meta)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_screening_report.html",
                                            prefix="acoustic_")
        os.close(fd)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✓  HTML report saved → {output_path}")

    if auto_open:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
        print("  ✓  Browser tab opened.")

    return output_path


# ═════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE  (v3 — wires everything together)
# ═════════════════════════════════════════════════════════════════════

class InferencePipeline:
    """
    Full pipeline v3: any audio source → terminal report + HTML dashboard.

    Auto-domain intelligence:
      • YouTube URL → domain=1.0 (Vlog) unless overridden
      • Local file  → domain=0.0 (Clinical) unless overridden

    Parameters
    ----------
    checkpoints_dir : str   path to folder with all checkpoint files
    domain          : float | None
        If None (default), domain is inferred automatically from the source.
        Pass 0.0 or 1.0 to override auto-detection.
    auto_open_browser : bool   open the HTML dashboard in a browser (default True)
    """

    def __init__(self, checkpoints_dir: str = "checkpoints",
                 domain: float = None,
                 auto_open_browser: bool = True):
        print("\n" + "─" * 70)
        print("  Initialising InferencePipeline v4…")
        print("─" * 70)

        self._checkpoints_dir  = checkpoints_dir
        self._forced_domain    = domain       # None = auto
        self._auto_open        = auto_open_browser

        # Loader is always created; FeatureExtractor is created per-run
        # because the domain may change between runs (auto-detection).
        self.loader = ModelLoader(checkpoints_dir)
        self.engine = InferenceEngine(self.loader)
        self.agent  = None   # created lazily with correct domain

        print("  ✓  InferencePipeline v4 ready\n")

    def _get_extractor(self, domain: float):
        """Return a FeatureExtractor for the given domain tag."""
        from real_time_feature_extractor import FeatureExtractor
        scaler_path = os.path.join(self._checkpoints_dir, "scaler_v3.pkl")
        extractor = FeatureExtractor(scaler_path=scaler_path, domain=domain)
        return extractor

    def run(self, source, output_html: str = None) -> dict:
        """
        Run the full pipeline.

        Parameters
        ----------
        source      : str or (np.ndarray, int)
            File path (WAV/MP3/FLAC/NPY), YouTube URL,
            or a (audio_array, sample_rate) tuple.
        output_html : str or None
            Optional explicit path for the HTML report.

        Returns
        -------
        report : dict
        """
        if isinstance(source, tuple):
            return self.run_array(*source, output_html=output_html)

        # ── Auto domain detection ──────────────────────────────────
        if self._forced_domain is not None:
            domain = self._forced_domain
            print(f"  [Domain] Manual override → domain = {domain}")
        else:
            domain = detect_domain(source)

        # ── Source metadata ────────────────────────────────────────
        is_youtube = isinstance(source, str) and (
            source.startswith("http://") or source.startswith("https://"))

        if is_youtube:
            print("  Fetching YouTube metadata…", end=" ", flush=True)
            yt_meta = _extract_youtube_metadata(source)
            print(f"✓  \"{yt_meta['title'][:50]}\"")
            source_meta = {
                "source_type" : "youtube",
                "yt_title"    : yt_meta["title"],
                "yt_channel"  : yt_meta["channel"],
                "yt_thumbnail": yt_meta["thumbnail_url"],
                "yt_url"      : source,
                "yt_duration" : yt_meta["duration_str"],
                "domain_tag"  : domain,
            }
        else:
            fname  = os.path.basename(str(source))
            ext    = os.path.splitext(fname)[1].lstrip(".").upper() or "AUDIO"
            source_meta = {
                "source_type" : "file",
                "filename"    : fname,
                "file_format" : ext,
                "domain_tag"  : domain,
            }

        print(f"  Source    : {source if len(str(source)) < 60 else str(source)[:57]+'…'}")
        print("  Extracting features…", end=" ", flush=True)

        extractor = self._get_extractor(domain)
        result = extractor.from_file(source)
        n_chunks = len(result.get("chunks", []))
        print(f"✓  ({result['n_frames']} frames, "
              f"{result['duration_sec']:.1f}s, {n_chunks} TAN windows)")

        report = self._run_from_result(result, source_meta, output_html)
        return report

    def run_array(self, audio: np.ndarray, sr: int,
                   domain: float = 0.0,
                   output_html: str = None) -> dict:
        print("  Extracting features from array…", end=" ", flush=True)
        extractor = self._get_extractor(domain)
        result = extractor.from_array(audio, sr)
        n_chunks = len(result.get("chunks", []))
        print(f"✓  ({result['n_frames']} frames, "
              f"{result['duration_sec']:.1f}s, {n_chunks} TAN windows)")
        source_meta = {"source_type": "file", "filename": "array_input",
                       "file_format": "RAW", "domain_tag": domain}
        return self._run_from_result(result, source_meta, output_html)

    def _run_from_result(self, result: dict,
                          source_meta: dict,
                          output_html: str = None) -> dict:
        print("  Running ensemble (TAN×{} + LightGBM)…".format(
            len(result.get("chunks", [1]))), end=" ", flush=True)

        prediction = self.engine.predict_from_features(
            enriched     = result["enriched"],
            dl_input     = result["dl_input"],
            dl_mask      = result["dl_mask"],
            chunks       = result.get("chunks"),
            chunk_masks  = result.get("chunk_masks"),
            chunk_starts = result.get("chunk_starts"),
        )
        print(f"✓  prob={prediction['ensemble_prob']:.4f} → {prediction['label']}")

        agent = AgentLayer(self.loader.scaler)
        report = agent.generate_report(
            prediction         = prediction,
            feat_norm          = result["feat_norm"],
            duration_sec       = result["duration_sec"],
            segment_feat_norms = result.get("segment_feat_norms"),
            chunk_starts       = result.get("chunk_starts"),
            fps                = result.get("fps", 100.0),
        )

        # ── HTML Dashboard ─────────────────────────────────────────
        save_and_open_report(
            report      = report,
            source_meta = source_meta,
            output_path = output_html,
            auto_open   = self._auto_open,
        )

        return report


# ═════════════════════════════════════════════════════════════════════
# COMMAND-LINE ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Depression screening v3 — Auto-Domain + Visual Dashboard."
    )
    parser.add_argument("audio",
        help="Path to WAV/MP3/FLAC/NPY file, or a YouTube URL.")
    parser.add_argument("--checkpoints", "-c", default="checkpoints",
        help="Path to checkpoints folder (default: ./checkpoints)")
    parser.add_argument("--domain", "-d", type=float, default=None,
        help="Force domain tag: 0.0=clinical, 1.0=vlog. "
             "Default: auto-detect from source.")
    parser.add_argument("--output-html", "-o", default=None,
        help="Save HTML report to this path (default: auto temp file).")
    parser.add_argument("--no-browser", action="store_true",
        help="Do not open the browser automatically.")
    parser.add_argument("--json", "-j", action="store_true",
        help="Also save report as JSON alongside the audio file.")
    args = parser.parse_args()

    pipeline = InferencePipeline(
        checkpoints_dir   = args.checkpoints,
        domain            = args.domain,
        auto_open_browser = not args.no_browser,
    )
    report = pipeline.run(args.audio, output_html=args.output_html)

    if args.json:
        base    = os.path.splitext(args.audio)[0]
        outpath = base + "_screening_report.json"
        def to_python(obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return obj
        with open(outpath, "w") as f:
            json.dump(report, f, indent=2, default=to_python)
        print(f"  Report saved → {outpath}")

    return report


if __name__ == "__main__":
    main()
