# ══════════════════════════════════════════════════════════════════════
# real_time_feature_extractor.py  ·  v2 — Full-Audio Edition
#
# UPGRADES OVER v1
# ────────────────
# • Universal ingestion  : WAV, MP3, FLAC, NPY (raw feature arrays)
# • YouTube integration  : pass a YouTube URL, audio is streamed via yt-dlp
# • Sliding-window DL    : instead of truncating to 3 s, the entire audio
#                          is sliced into overlapping 3-s chunks; every
#                          chunk is stored so InferencePipeline can run
#                          TAN on each and average the results.
# • Per-segment features : from_file / from_array now also return
#                          'segment_feat_norms' — a list of [T_chunk, 23]
#                          arrays, one per 3-s window (used by the engine
#                          to identify *when* symptoms were strongest).
#
# PIPELINE (stages unchanged — still bit-for-bit training-compatible)
# ────────────────────────────────────────────────────────────────────
#   Stage 1 : OpenSMILE eGeMAPSv01b LLD extraction  → [T, 23]   raw feat
#   Stage 2 : StandardScaler.transform()             → [T, 23]   feat_norm
#   Stage 3 : Append domain tag                      → [T, 24]   feat_norm_dl
#   Stage 4 : Pad / truncate to MAX_SEQ_LEN=300      → [300, 24] dl_input  (global)
#   Stage 4b: Sliding-window chunks                  → list of [300, 24]   (new)
#   Stage 5 : 598-dim enriched vector                → [598]     for LightGBM
# ══════════════════════════════════════════════════════════════════════

import os
import io
import pickle
import tempfile
import warnings
import subprocess

import numpy as np
import soundfile as sf
import opensmile
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match training notebook exactly)
# ─────────────────────────────────────────────────────────────────────
N_FEATURES          = 23
N_FEATURES_DL       = 24
MAX_SEQ_LEN         = 300          # frames  ≈ 3 s at ~100 fps
DOMAIN_TAG_CLINICAL = 0.0
DOMAIN_TAG_VLOG     = 1.0
TARGET_SR           = 16000

EGEMAPS_LLD_FEATURES = [
    'Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3',
    'slope0-500_sma3', 'slope500-1500_sma3', 'spectralFlux_sma3',
    'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'mfcc4_sma3',
    'F0semitoneFrom27.5Hz_sma3nz', 'jitterLocal_sma3nz',
    'shimmerLocaldB_sma3nz', 'HNRdBACF_sma3nz',
    'logRelF0-H1-H2_sma3nz', 'logRelF0-H1-A3_sma3nz',
    'F1frequency_sma3nz', 'F1bandwidth_sma3nz',
    'F1amplitudeLogRelF0_sma3nz', 'F2frequency_sma3nz',
    'F2amplitudeLogRelF0_sma3nz', 'F3frequency_sma3nz',
    'F3amplitudeLogRelF0_sma3nz',
]


# ═════════════════════════════════════════════════════════════════════
# UNIVERSAL AUDIO LOADING
# ═════════════════════════════════════════════════════════════════════

def _load_audio_from_youtube(url: str) -> tuple:
    """
    Stream audio from a YouTube URL using yt-dlp.
    Returns (audio_array: float32 mono, sr: int).
    Requires: pip install yt-dlp
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise ImportError(
            "yt-dlp is required for YouTube input.\n"
            "Install with:  pip install yt-dlp"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "yt_audio.wav")
        ydl_opts = {
            'format': 'best',  # <--- CHANGED THIS: Grabs the video, bypasses the audio block
            'outtmpl': os.path.join(tmpdir, 'yt_audio.%(ext)s'),
            'cookiefile': 'cookies.txt',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp writes the file; find it
        wav_files = [f for f in os.listdir(tmpdir) if f.endswith('.wav')]
        if not wav_files:
            raise RuntimeError("yt-dlp did not produce a WAV file.")
        audio, sr = sf.read(os.path.join(tmpdir, wav_files[0]),
                            always_2d=False, dtype='float32')
    return audio, sr


def _load_audio_universal(source) -> tuple:
    """
    Load audio from any source:
      • str ending in .npy  → pre-extracted feature array  [T, 23]  (bypass Stage 1)
      • str starting with http(s)://  → YouTube URL
      • str (file path)     → WAV / MP3 / FLAC via soundfile + fallback to pydub
      • (np.ndarray, int)   → (audio_array, sample_rate)

    Returns one of:
        ('array',  audio_np, sr)          normal audio
        ('npy',    feat_norm_raw, None)   pre-extracted features, skip stage 1
    """
    if isinstance(source, tuple):
        audio, sr = source
        return ('array', audio.astype(np.float32), sr)

    if isinstance(source, np.ndarray):
        raise ValueError(
            "Pass (array, sr) as a tuple when providing a numpy array directly."
        )

    if isinstance(source, str):
        # YouTube URL
        if source.startswith("http://") or source.startswith("https://"):
            audio, sr = _load_audio_from_youtube(source)
            return ('array', audio, sr)

        # Pre-extracted .npy feature file
        if source.lower().endswith('.npy'):
            feat = np.load(source).astype(np.float32)
            if feat.ndim == 1:
                # Might be a flat enriched vector — cannot use; need frame-level
                raise ValueError(
                    "NPY file appears to be a 1-D vector. "
                    "Expected a [T, 23] frame-level feature array."
                )
            if feat.shape[1] != N_FEATURES:
                raise ValueError(
                    f"NPY file has {feat.shape[1]} features per frame, "
                    f"expected {N_FEATURES}. Ensure it was extracted with "
                    f"OpenSMILE eGeMAPSv01b LLD."
                )
            return ('npy', feat, None)

        # Regular audio file (WAV, MP3, FLAC, OGG, …)
        try:
            audio, sr = sf.read(source, always_2d=False, dtype='float32')
            return ('array', audio, sr)
        except Exception as sf_err:
            # Fallback: try pydub (handles MP3 better on some systems)
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(source)
                seg = seg.set_channels(1).set_frame_rate(TARGET_SR)
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                samples /= 32768.0
                return ('array', samples, TARGET_SR)
            except ImportError:
                raise ImportError(
                    f"soundfile failed to read '{source}' ({sf_err}). "
                    "For MP3 support install pydub:  pip install pydub"
                )
            except Exception as pydub_err:
                raise RuntimeError(
                    f"Could not load audio from '{source}'.\n"
                    f"  soundfile error : {sf_err}\n"
                    f"  pydub error     : {pydub_err}"
                )

    raise TypeError(f"Unsupported source type: {type(source)}")


# ═════════════════════════════════════════════════════════════════════
# INTERNAL PIPELINE FUNCTIONS  (bit-for-bit identical to v1 / training)
# ═════════════════════════════════════════════════════════════════════

def _build_smile() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_channels=1,
    )


# Stage 1
def _extract_raw_features(audio: np.ndarray,
                           sr: int,
                           smile: opensmile.Smile) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        except ImportError:
            raise ImportError(
                "librosa is required to resample audio that isn't 16 kHz.\n"
                "Install with:  pip install librosa"
            )
        sr = TARGET_SR

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmpname = tmp.name
    try:
        df = smile.process_file(tmpname)
    finally:
        os.unlink(tmpname)

    feat = df.values.astype(np.float32)
    if feat.shape[1] != N_FEATURES:
        raise RuntimeError(
            f"OpenSMILE returned {feat.shape[1]} features, expected {N_FEATURES}."
        )
    return feat


# Stage 2
def _normalize(feat: np.ndarray, scaler) -> np.ndarray:
    return scaler.transform(feat).astype(np.float32)


# Stage 3
def _append_domain_tag(feat_norm: np.ndarray, domain: float) -> np.ndarray:
    T = feat_norm.shape[0]
    dtag = np.full((T, 1), domain, dtype=np.float32)
    return np.concatenate([feat_norm, dtag], axis=1)


# Stage 4 — global (same as v1)
def _pad_or_truncate(feat_norm_dl: np.ndarray,
                      max_len: int = MAX_SEQ_LEN):
    T, D     = feat_norm_dl.shape
    orig_len = T
    if T >= max_len:
        padded = feat_norm_dl[:max_len].copy()
        mask   = np.ones(max_len, dtype=bool)
    else:
        padded = np.zeros((max_len, D), dtype=np.float32)
        padded[:T] = feat_norm_dl
        mask = np.zeros(max_len, dtype=bool)
        mask[:T] = True
    return padded, mask, orig_len


# Stage 4b — SLIDING WINDOW (new)
def _sliding_window_chunks(feat_norm_dl: np.ndarray,
                             window: int = MAX_SEQ_LEN,
                             hop: int = 150):
    """
    Slice the full feature sequence into overlapping windows.

    Parameters
    ----------
    feat_norm_dl : [T, 24]  full normalised + domain-tagged sequence
    window       : int      chunk size in frames (300 ≈ 3 s)
    hop          : int      hop in frames between windows (150 ≈ 50 % overlap)

    Returns
    -------
    chunks      : list of np.ndarray  each [300, 24], zero-padded if short
    chunk_masks : list of np.ndarray  each [300] bool
    chunk_starts: list of int         frame index where each chunk begins
    """
    T, D = feat_norm_dl.shape
    chunks, chunk_masks, chunk_starts = [], [], []

    start = 0
    while start < T:
        end     = start + window
        segment = feat_norm_dl[start:min(end, T)]
        seg_len = segment.shape[0]

        padded = np.zeros((window, D), dtype=np.float32)
        padded[:seg_len] = segment
        mask   = np.zeros(window, dtype=bool)
        mask[:seg_len] = True

        chunks.append(padded)
        chunk_masks.append(mask)
        chunk_starts.append(start)

        if end >= T:
            break
        start += hop

    return chunks, chunk_masks, chunk_starts


# Stage 5 helpers (identical to v1 / training)
def _compute_delta(feat: np.ndarray, order: int = 1) -> np.ndarray:
    delta = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        p = max(0, t - 1)
        n = min(feat.shape[0] - 1, t + 1)
        delta[t] = (feat[n] - feat[p]) / 2.0
    return _compute_delta(delta, order=1) if order == 2 else delta


def _segment_stats(feat: np.ndarray, n: int = 5) -> np.ndarray:
    T, D = feat.shape
    seg  = max(1, T // n)
    vecs = []
    for i in range(n):
        s     = i * seg
        e     = s + seg if i < n - 1 else T
        seg_f = feat[s:e]
        vecs.extend([seg_f.mean(0), seg_f.std(0) + 1e-8])
    return np.concatenate(vecs)


def _crossing_rate(feat: np.ndarray) -> np.ndarray:
    mu    = feat.mean(0, keepdims=True)
    signs = np.sign(feat - mu)
    return np.abs(np.diff(signs, axis=0)).sum(0) / max(feat.shape[0] - 1, 1)


def _autocorr(feat: np.ndarray, lags=(1, 5, 10)) -> np.ndarray:
    T, D = feat.shape
    res  = []
    for lag in lags:
        if T > lag:
            a   = feat[:-lag] - feat[:-lag].mean(0)
            b   = feat[lag:]  - feat[lag:].mean(0)
            den = (np.std(feat[:-lag], 0) *
                   np.std(feat[lag:],  0) * (T - lag))
            res.append((a * b).sum(0) / np.maximum(den, 1e-8))
        else:
            res.append(np.zeros(D))
    return np.concatenate(res)


# Stage 5
def _build_enriched_vector(feat_norm: np.ndarray) -> np.ndarray:
    f  = feat_norm
    d1 = _compute_delta(f, order=1)
    d2 = _compute_delta(f, order=2)
    vec = np.concatenate([
        f.mean(0), f.std(0), f.min(0), f.max(0),
        np.median(f, 0), skew(f, 0), kurtosis(f, 0),
        d1.mean(0), d1.std(0), d2.mean(0), d2.std(0),
        _segment_stats(f, n=5),
        _crossing_rate(f),
        np.percentile(f, 90, 0) - np.percentile(f, 10, 0),
        _autocorr(f, lags=(1, 5, 10)),
    ])
    return np.nan_to_num(vec, nan=0., posinf=0., neginf=0.).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════
# MAIN CLASS — FeatureExtractor  v2
# ═════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Universal feature extractor — now accepts WAV, MP3, FLAC, NPY, or
    a YouTube URL and runs the full sliding-window pipeline.

    Parameters
    ----------
    scaler_path : str    path to scaler_v3.pkl
    domain      : float  0.0 = clinical/E-DAIC (default), 1.0 = D-Vlog
    window      : int    sliding window size in frames (default 300 = 3 s)
    hop         : int    hop between windows in frames  (default 150 = 1.5 s)

    What's new in v2
    ----------------
    The returned dict now contains:
        'chunks'        : list of np.ndarray  each [300, 24] — one per window
        'chunk_masks'   : list of np.ndarray  each [300] bool
        'chunk_starts'  : list of int         start frame index per window
        'segment_feat_norms': list of [T_chunk, 23] — raw normalised per window
        (all v1 keys are still present and unchanged)
    """

    def __init__(self,
                 scaler_path: str,
                 domain: float = DOMAIN_TAG_CLINICAL,
                 window: int   = MAX_SEQ_LEN,
                 hop: int      = 150):
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.smile  = _build_smile()
        self.domain = domain
        self.window = window
        self.hop    = hop
        print("FeatureExtractor v2 ready.")
        print(f"  Scaler   : {scaler_path}")
        print(f"  Domain   : {domain} "
              f"({'clinical/E-DAIC' if domain == 0.0 else 'vlog/D-Vlog'})")
        print(f"  Window   : {window} frames | Hop : {hop} frames")

    # ── Public entry points ──────────────────────────────────────────

    def from_file(self, audio_path: str) -> dict:
        """
        Extract from a file path (WAV / MP3 / FLAC / NPY) or YouTube URL.
        """
        return self._run(audio_path)

    def from_array(self, audio: np.ndarray, sr: int) -> dict:
        """
        Extract from a numpy audio array.
        """
        return self._run((audio, sr))

    def from_bytes(self, audio_bytes: bytes, sr: int) -> dict:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        return self.from_array(audio, sr)

    def get_dl_tensor(self, source, sr: int = None):
        """Returns (x [1,300,24], mask [1,300]) for the FIRST chunk."""
        import torch
        if isinstance(source, str):
            result = self.from_file(source)
        else:
            if sr is None:
                raise ValueError("sr required when passing a numpy array.")
            result = self.from_array(source, sr)
        x    = torch.FloatTensor(result['dl_input']).unsqueeze(0)
        mask = torch.BoolTensor(result['dl_mask']).unsqueeze(0)
        return x, mask

    # ── Core pipeline ────────────────────────────────────────────────

    def _run(self, source) -> dict:
        kind, data, sr = _load_audio_universal(source)

        if kind == 'npy':
            # Pre-extracted [T, 23] normalised features — skip Stage 1 & 2
            feat_norm    = data
            duration_sec = feat_norm.shape[0] / 100.0   # ~100 fps assumption
        else:
            audio = data
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            duration_sec = len(audio) / sr

            # Stage 1
            feat_raw  = _extract_raw_features(audio, sr, self.smile)
            # Stage 2
            feat_norm = _normalize(feat_raw, self.scaler)

        # Stage 3
        feat_norm_dl = _append_domain_tag(feat_norm, self.domain)

        # Stage 4 — global (unchanged; used by LightGBM path + fallback TAN)
        dl_input, dl_mask, n_frames = _pad_or_truncate(
            feat_norm_dl, self.window)

        # Stage 4b — sliding window (new)
        chunks, chunk_masks, chunk_starts = _sliding_window_chunks(
            feat_norm_dl, window=self.window, hop=self.hop)

        # Build per-chunk normalised feat_norm slices (for temporal reasoning)
        fps = feat_norm.shape[0] / max(duration_sec, 1e-6)
        segment_feat_norms = []
        for start_f, chunk_mask in zip(chunk_starts, chunk_masks):
            real_len = int(chunk_mask.sum())
            end_f    = start_f + real_len
            segment_feat_norms.append(feat_norm[start_f:end_f])

        # Stage 5 — enriched vector (global, for LightGBM)
        enriched = _build_enriched_vector(feat_norm)

        return {
            # ── v1 keys (unchanged) ──────────────────────────────────
            'feat_norm'          : feat_norm,
            'feat_norm_dl'       : feat_norm_dl,
            'dl_input'           : dl_input,
            'dl_mask'            : dl_mask,
            'enriched'           : enriched,
            'n_frames'           : n_frames,
            'duration_sec'       : duration_sec,
            # ── v2 keys (new) ────────────────────────────────────────
            'chunks'             : chunks,          # list of [300, 24]
            'chunk_masks'        : chunk_masks,     # list of [300] bool
            'chunk_starts'       : chunk_starts,    # list of int (frame idx)
            'segment_feat_norms' : segment_feat_norms,  # list of [T_i, 23]
            'fps'                : fps,
        }


# ═════════════════════════════════════════════════════════════════════
# REAL-TIME RECORDER (unchanged from v1)
# ═════════════════════════════════════════════════════════════════════

class RealTimeRecorder:
    def __init__(self, scaler_path: str, duration_sec: int = 30,
                 sr: int = TARGET_SR, domain: float = DOMAIN_TAG_CLINICAL):
        self.extractor    = FeatureExtractor(scaler_path, domain)
        self.duration_sec = duration_sec
        self.sr           = sr

    def record_and_extract(self) -> dict:
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("pip install sounddevice")
        print(f"\nRecording {self.duration_sec}s at {self.sr} Hz… Speak now → ",
              end='', flush=True)
        audio = sd.rec(int(self.duration_sec * self.sr),
                       samplerate=self.sr, channels=1, dtype='float32')
        sd.wait()
        audio = audio.squeeze()
        print("Done.")
        return self.extractor.from_array(audio, self.sr)
