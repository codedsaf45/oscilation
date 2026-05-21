"""
realtime_spectrogram_45.py  (v2: dominant frequency tracking)
==============================================================
- 시계열 + PSD(현재 순간) + 스펙트로그램(누적)
- Dominant frequency 실시간 표시 (peak Hz, peak dB)
- DC + 저주파 제외 옵션 (baseline 잔류 영향 제거)
- Top-N peaks 콘솔/플롯 표시

조작:
  [0-9]  센서 ID 선택  (두자리는 빠르게 두번)
  [A]    전체 평균 토글
  [R]    Baseline 재캘리브
  [F]    DC 차단 freq 토글 (0 / 1 / 5 / 10 Hz)
  [Q]    종료
"""

import serial
import time
import threading
import queue
import struct
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================================================
# 설정
# =========================================================
CFG = dict(
    port      = "COM8",
    baudrate  = 115200,

    buffer_sec   = 4.0,
    fft_window   = 512,        # 분해능 ↑ (256 → 512)
    fft_hop      = 128,
    spec_history = 200,
    expected_fs  = 1000.0,

    display_sensor_id = 22,
    show_average      = False,
    plot_refresh_ms   = 50,

    # ── peak 탐색 설정 ──
    dc_cutoff_hz   = 1.0,    # 이 주파수 이하는 peak 후보에서 제외
    top_n_peaks    = 3,
    peak_min_db    = -80.0,  # 이 dB 미만이면 "신호 없음" 처리

    autoscale_time = True,
)

# =========================================================
# 프로토콜
# =========================================================
HEADER = 0xAA
FOOTER = 0x55
BYTES_PER_SENSOR = 13
N_SENSORS = 45


class TMRSerialReader(threading.Thread):
    def __init__(self, port, baudrate, data_queue, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.ser = None
        self.n_ok = 0
        self.n_err = 0

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.ser.reset_input_buffer()
            print(f"  ✓ Serial open: {self.port}")
            while not self.stop_event.is_set():
                if not self._sync_header():
                    continue
                self._read_packet()
        except Exception as e:
            print(f"  ✗ Serial error: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def _sync_header(self):
        while not self.stop_event.is_set():
            b = self.ser.read(1)
            if not b:
                return False
            if b[0] == HEADER:
                return True
        return False

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n and not self.stop_event.is_set():
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                continue
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

    def _read_packet(self):
        count_b = self._read_exact(1)
        if count_b is None:
            return
        count = count_b[0]
        if count == 0 or count > 100:
            self.n_err += 1
            return
        payload = self._read_exact(count * BYTES_PER_SENSOR)
        if payload is None:
            self.n_err += 1
            return
        tail = self._read_exact(2)
        if tail is None:
            self.n_err += 1
            return
        recv_chk = tail[0]
        footer = tail[1]
        if footer != FOOTER:
            self.n_err += 1
            return
        calc = count
        for byte in payload:
            calc ^= byte
        if calc != recv_chk:
            self.n_err += 1
            return

        xs = np.zeros(N_SENSORS, dtype=np.float32)
        ok = 0
        for i in range(count):
            off = i * BYTES_PER_SENSOR
            sid = payload[off]
            if sid < N_SENSORS:
                xs[sid] = struct.unpack_from("<f", payload, off + 1)[0]
                ok += 1
        if ok < N_SENSORS * 0.5:
            self.n_err += 1
            return
        self.n_ok += 1

        try:
            self.data_queue.put_nowait((time.perf_counter(), xs))
        except queue.Full:
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait((time.perf_counter(), xs))
            except queue.Empty:
                pass


# =========================================================
# Peak 탐색 유틸
# =========================================================
def find_peaks_simple(y, min_distance=2):
    """간단한 local maxima 탐색 (scipy 없이)."""
    peaks = []
    n = len(y)
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(i)
    if not peaks:
        return np.array([], dtype=int)
    peaks = np.array(peaks, dtype=int)
    # 최소 간격 적용 (높은 것부터)
    if min_distance > 1:
        order = np.argsort(-y[peaks])
        kept = []
        used = np.zeros(n, dtype=bool)
        for idx in peaks[order]:
            lo = max(0, idx - min_distance)
            hi = min(n, idx + min_distance + 1)
            if used[lo:hi].any():
                continue
            kept.append(idx)
            used[lo:hi] = True
        peaks = np.array(sorted(kept), dtype=int)
    return peaks


def parabolic_interp(y, i):
    """3점 포물선 보간으로 peak의 fractional bin 추정."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(i)
    return i + 0.5 * (y0 - y2) / denom


# =========================================================
# Spectrogram App
# =========================================================
class SpectrogramApp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fs_est = float(cfg["expected_fs"])
        self.display_sid = cfg["display_sensor_id"]
        self.show_avg = cfg["show_average"]

        self.maxlen = int(self.fs_est * cfg["buffer_sec"])
        self.t_buf = deque(maxlen=self.maxlen)
        self.x_buf = deque(maxlen=self.maxlen)

        self.baseline = np.zeros(N_SENSORS, dtype=np.float32)
        self.baseline_n = 0
        self.baseline_target = 200
        self.baseline_ready = False

        self.fps_window = deque(maxlen=200)
        self.fs_current = 0.0

        n_freq = cfg["fft_window"] // 2 + 1
        self.spec_history = np.full((n_freq, cfg["spec_history"]),
                                     -120.0, dtype=np.float32)
        self.dom_freq_history = np.full(cfg["spec_history"],
                                         np.nan, dtype=np.float32)
        self.samples_since_last_fft = 0

        # 최신 PSD / peak 정보
        self.latest_psd_db = None     # shape (n_freq,)
        self.latest_freqs = None
        self.latest_peaks = []        # [(freq, db), ...]
        self.dc_cutoff_hz = cfg["dc_cutoff_hz"]
        self._cutoff_options = [0.0, 1.0, 5.0, 10.0]

        self.data_queue = queue.Queue(maxsize=200)
        self.stop_event = threading.Event()
        self.reader = TMRSerialReader(cfg["port"], cfg["baudrate"],
                                       self.data_queue, self.stop_event)

        self.window_func = np.hanning(cfg["fft_window"]).astype(np.float32)
        self.window_norm = float(np.sum(self.window_func ** 2))

        self._key_accum = ""

    def _update_baseline(self, xs):
        if self.baseline_ready:
            return
        self.baseline = (self.baseline * self.baseline_n + xs) / (self.baseline_n + 1)
        self.baseline_n += 1
        if self.baseline_n >= self.baseline_target:
            self.baseline_ready = True
            print(f"  ✓ Baseline ready ({self.baseline_n} frames)")

    def _consume(self):
        while True:
            try:
                t, xs = self.data_queue.get_nowait()
            except queue.Empty:
                break

            self._update_baseline(xs)
            xs_corr = xs - self.baseline

            if self.show_avg:
                val = float(np.mean(xs_corr))
            else:
                val = float(xs_corr[self.display_sid])

            self.t_buf.append(t)
            self.x_buf.append(val)
            self.fps_window.append(t)
            self.samples_since_last_fft += 1

        if len(self.fps_window) >= 2:
            span = self.fps_window[-1] - self.fps_window[0]
            if span > 0:
                self.fs_current = (len(self.fps_window) - 1) / span

        if self.fs_current > 0:
            target = int(self.fs_current * self.cfg["buffer_sec"])
            target = max(target, self.cfg["fft_window"] * 4)
            if target != self.t_buf.maxlen:
                self.t_buf = deque(self.t_buf, maxlen=target)
                self.x_buf = deque(self.x_buf, maxlen=target)

    def _compute_spectrogram_slice(self):
        N = self.cfg["fft_window"]
        if len(self.x_buf) < N:
            return None, None
        data = np.array(list(self.x_buf)[-N:], dtype=np.float32)
        data = data - data.mean()
        data *= self.window_func
        spec = np.fft.rfft(data)
        psd = (np.abs(spec) ** 2) / (self.window_norm + 1e-12)
        psd_db = 10.0 * np.log10(psd + 1e-12).astype(np.float32)

        if self.fs_current > 0:
            freqs = np.fft.rfftfreq(N, d=1.0 / self.fs_current)
        else:
            freqs = np.arange(len(psd_db), dtype=np.float32)
        return psd_db, freqs

    def _find_dominant(self, psd_db, freqs):
        """DC 차단 후 top-N peak 반환."""
        if psd_db is None or freqs is None or len(psd_db) < 4:
            return []
        mask = freqs >= self.dc_cutoff_hz
        if not mask.any():
            return []

        psd_masked = psd_db.copy()
        psd_masked[~mask] = -1e9   # DC 영역 invalidate

        peaks = find_peaks_simple(psd_masked, min_distance=2)
        # peak_min_db 필터
        peaks = peaks[psd_masked[peaks] >= self.cfg["peak_min_db"]]
        if len(peaks) == 0:
            return []
        # 강도 내림차순 정렬
        order = np.argsort(-psd_masked[peaks])
        peaks = peaks[order][:self.cfg["top_n_peaks"]]

        results = []
        for p in peaks:
            # 포물선 보간으로 더 정확한 주파수
            p_frac = parabolic_interp(psd_db, int(p))
            if 0 <= p_frac < len(freqs) - 1:
                lo = int(np.floor(p_frac))
                frac = p_frac - lo
                freq_est = freqs[lo] * (1 - frac) + freqs[lo + 1] * frac
            else:
                freq_est = float(freqs[int(p)])
            results.append((float(freq_est), float(psd_db[int(p)])))
        return results

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    def setup_plot(self):
        plt.style.use("default")
        self.fig, axes = plt.subplots(
            3, 1, figsize=(11, 9),
            gridspec_kw={"height_ratios": [1, 1, 2]},
        )
        self.ax_time, self.ax_psd, self.ax_spec = axes
        self.fig.subplots_adjust(hspace=0.45, bottom=0.07, top=0.94)

        # 시계열
        self.ax_time.set_title(self._title_str())
        self.ax_time.set_xlabel("time (s)")
        self.ax_time.set_ylabel("ΔBz (mT)")
        self.ax_time.grid(True, alpha=0.3)
        (self.line_time,) = self.ax_time.plot([], [], lw=1.0, color="C0")

        # PSD (현재 순간)
        self.ax_psd.set_title("Current PSD")
        self.ax_psd.set_xlabel("freq (Hz)")
        self.ax_psd.set_ylabel("dB")
        self.ax_psd.grid(True, alpha=0.3)
        (self.line_psd,) = self.ax_psd.plot([], [], lw=1.0, color="C2")
        # peak 마커
        self.peak_scatter = self.ax_psd.scatter(
            [], [], s=60, marker="v",
            facecolor="red", edgecolor="black", zorder=5,
        )
        self.peak_text = self.ax_psd.text(
            0.98, 0.95, "", transform=self.ax_psd.transAxes,
            ha="right", va="top", fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="gray", alpha=0.9),
        )

        # 스펙트로그램
        self.spec_im = self.ax_spec.imshow(
            self.spec_history,
            origin="lower", aspect="auto",
            cmap="magma", vmin=-60, vmax=20,
            interpolation="nearest",
        )
        self.ax_spec.set_xlabel("time slice (recent →)")
        self.ax_spec.set_ylabel("freq (Hz)")
        self.ax_spec.set_title("STFT (dB)  —  red dot: dominant freq")
        self.cb = self.fig.colorbar(self.spec_im, ax=self.ax_spec,
                                     pad=0.01, label="dB")
        # 스펙트로그램 위 dominant freq 추적 라인
        (self.dom_line,) = self.ax_spec.plot(
            [], [], "o-", color="lime", ms=3, lw=0.8, alpha=0.7,
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: self._on_quit())

    def _title_str(self):
        tag = "AVG(45)" if self.show_avg else f"sensor #{self.display_sid}"
        if self.latest_peaks:
            dom_f, dom_db = self.latest_peaks[0]
            dom_str = f"  dominant: {dom_f:6.2f} Hz ({dom_db:+5.1f} dB)"
        else:
            dom_str = "  dominant: --"
        return (f"{tag}   fs = {self.fs_current:6.1f} Hz   "
                f"buf = {len(self.x_buf):5d}   "
                f"DC-cut = {self.dc_cutoff_hz:.1f} Hz" + dom_str)

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == "q":
            self._on_quit()
        elif k == "r":
            print("  ↻ Recalibrating baseline...")
            self.baseline[:] = 0
            self.baseline_n = 0
            self.baseline_ready = False
        elif k == "a":
            self.show_avg = not self.show_avg
            print(f"  → AVG mode = {self.show_avg}")
            self.t_buf.clear()
            self.x_buf.clear()
        elif k == "f":
            # DC cutoff 순환
            idx = self._cutoff_options.index(self.dc_cutoff_hz) \
                if self.dc_cutoff_hz in self._cutoff_options else 0
            idx = (idx + 1) % len(self._cutoff_options)
            self.dc_cutoff_hz = self._cutoff_options[idx]
            print(f"  → DC cutoff = {self.dc_cutoff_hz} Hz")
        elif k and k.isdigit():
            self._key_accum += k
            if len(self._key_accum) >= 2 or int(self._key_accum) >= 5:
                try:
                    sid = int(self._key_accum)
                    if 0 <= sid < N_SENSORS:
                        self.display_sid = sid
                        self.show_avg = False
                        print(f"  → display sensor = {sid}")
                        self.t_buf.clear()
                        self.x_buf.clear()
                except ValueError:
                    pass
                self._key_accum = ""

    def _on_quit(self):
        self.stop_event.set()
        try:
            plt.close("all")
        except Exception:
            pass

    def animate(self, frame):
        self._consume()

        # ── 시계열 ──
        if len(self.x_buf) > 1:
            ts = np.array(self.t_buf)
            xs = np.array(self.x_buf)
            ts = ts - ts[-1]
            self.line_time.set_data(ts, xs)
            self.ax_time.set_xlim(ts[0], 0)
            if self.cfg["autoscale_time"]:
                m = max(0.01, np.max(np.abs(xs)))
                self.ax_time.set_ylim(-m * 1.2, m * 1.2)

        # ── 스펙트로그램 슬라이스 ──
        if self.samples_since_last_fft >= self.cfg["fft_hop"]:
            self.samples_since_last_fft = 0
            psd_db, freqs = self._compute_spectrogram_slice()
            if psd_db is not None:
                self.latest_psd_db = psd_db
                self.latest_freqs = freqs
                self.latest_peaks = self._find_dominant(psd_db, freqs)

                # 누적
                self.spec_history = np.roll(self.spec_history, -1, axis=1)
                self.spec_history[:, -1] = psd_db
                self.spec_im.set_data(self.spec_history)
                if self.fs_current > 0:
                    self.spec_im.set_extent(
                        [0, self.spec_history.shape[1],
                         float(freqs[0]), float(freqs[-1])]
                    )

                # dominant freq history
                self.dom_freq_history = np.roll(self.dom_freq_history, -1)
                self.dom_freq_history[-1] = (
                    self.latest_peaks[0][0] if self.latest_peaks else np.nan
                )
                xs_line = np.arange(len(self.dom_freq_history)) + 0.5
                valid = ~np.isnan(self.dom_freq_history)
                self.dom_line.set_data(xs_line[valid],
                                        self.dom_freq_history[valid])

                # PSD 갱신
                self.line_psd.set_data(freqs, psd_db)
                if len(freqs) > 1:
                    self.ax_psd.set_xlim(0, freqs[-1])
                ymax = float(np.max(psd_db)) + 5
                ymin = max(-100, float(np.min(psd_db)) - 5)
                self.ax_psd.set_ylim(ymin, ymax)

                # peak 마커
                if self.latest_peaks:
                    pf = np.array([p[0] for p in self.latest_peaks])
                    pd = np.array([p[1] for p in self.latest_peaks])
                    self.peak_scatter.set_offsets(np.column_stack([pf, pd]))
                    lines = []
                    for i, (f, d) in enumerate(self.latest_peaks):
                        prefix = "★" if i == 0 else " "
                        lines.append(f"{prefix} {f:6.2f} Hz  {d:+5.1f} dB")
                    self.peak_text.set_text("\n".join(lines))
                else:
                    self.peak_scatter.set_offsets(np.empty((0, 2)))
                    self.peak_text.set_text("(no peak)")

        self.ax_time.set_title(self._title_str())

    def run(self):
        print("=" * 60)
        print("  Realtime Spectrogram (45ch) — dominant frequency tracker")
        print("=" * 60)
        print("  [0-9] sensor | [A] AVG | [R] baseline")
        print("  [F] DC-cut toggle (0/1/5/10 Hz) | [Q] quit")
        print("=" * 60)

        self.reader.start()
        self.setup_plot()
        anim = FuncAnimation(
            self.fig, self.animate,
            interval=self.cfg["plot_refresh_ms"],
            blit=False, cache_frame_data=False,
        )
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            self.reader.join(timeout=1.0)


if __name__ == "__main__":
    app = SpectrogramApp(CFG)
    app.run()
