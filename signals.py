"""Generador de señales moduladas (banda base compleja I/Q) para ASK, FSK, PSK y QAM."""
import numpy as np

CLASES = ["ASK", "FSK", "PSK", "QAM"]
SPS = 8            # muestras por símbolo
N_SIMBOLOS = 128   # símbolos por señal
N_MUESTRAS = SPS * N_SIMBOLOS  # 1024 muestras por señal


def _normalizar(x):
    """Potencia promedio = 1."""
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def gen_ask(rng):
    M = rng.choice([2, 4])                       # OOK / 4-ASK
    niveles = np.linspace(0, 1, M)
    simbolos = rng.choice(niveles, N_SIMBOLOS)
    return np.repeat(simbolos, SPS).astype(complex)


def gen_fsk(rng):
    M = rng.choice([2, 4])                       # 2-FSK / 4-FSK
    desviacion = rng.uniform(0.04, 0.10)         # frecuencia normalizada (ciclos/muestra)
    freqs = (np.arange(M) - (M - 1) / 2) * 2 * desviacion
    simbolos = rng.choice(freqs, N_SIMBOLOS)
    f_inst = np.repeat(simbolos, SPS)
    fase = 2 * np.pi * np.cumsum(f_inst)         # fase continua
    return np.exp(1j * fase)


def gen_psk(rng):
    M = rng.choice([2, 4, 8])                    # BPSK / QPSK / 8-PSK
    fases = 2 * np.pi * rng.integers(0, M, N_SIMBOLOS) / M
    return np.repeat(np.exp(1j * fases), SPS)


def gen_qam(rng):
    M = rng.choice([16, 64])                     # 16-QAM / 64-QAM
    L = int(np.sqrt(M))
    niveles = np.arange(-(L - 1), L, 2)
    I = rng.choice(niveles, N_SIMBOLOS)
    Q = rng.choice(niveles, N_SIMBOLOS)
    return np.repeat(I + 1j * Q, SPS)


GENERADORES = {"ASK": gen_ask, "FSK": gen_fsk, "PSK": gen_psk, "QAM": gen_qam}


def agregar_ruido(x, snr_db, rng):
    """Ruido blanco gaussiano (AWGN) para una SNR dada en dB."""
    p_ruido = 10 ** (-snr_db / 10)
    ruido = np.sqrt(p_ruido / 2) * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    return x + ruido


def generar_senal(clase, snr_db, rng):
    x = GENERADORES[clase](rng)
    x = x * np.exp(1j * rng.uniform(0, 2 * np.pi))   # fase inicial aleatoria
    x = _normalizar(x)
    return agregar_ruido(x, snr_db, rng)


def generar_dataset(n_por_clase=1500, snr_min=-4, snr_max=20, semilla=42):
    """Devuelve X (N, 1024, 2) con canales I/Q, y (N,), snr (N,)."""
    rng = np.random.default_rng(semilla)
    X, y, snrs = [], [], []
    for idx, clase in enumerate(CLASES):
        for _ in range(n_por_clase):
            snr = rng.uniform(snr_min, snr_max)
            s = generar_senal(clase, snr, rng)
            X.append(np.stack([s.real, s.imag], axis=-1))
            y.append(idx)
            snrs.append(snr)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    snrs = np.array(snrs, dtype=np.float32)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], snrs[perm]
