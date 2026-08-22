## 1. Project Overview

This project investigates how well a simple first-order dynamic model can predict the current behavior of an ESP32 under controlled CPU and peripheral workloads.

The model is first identified from CPU activity and validated on unseen CPU workloads. Its transferability is then tested on UART, BLE, and ADC activity. When the model fails, the analysis focuses on identifying whether the main cause is the model parameters, the first-order structure, or the workload-input representation.

**Research Question**

When prediction fails, is the main cause the model parameters, the model structure, or the workload input?

---

## 2. Results at a Glance

| Area | Result | Main issue |
| --- | --- | --- |
| CPU | Good fit | Input timing must be represented explicitly |
| UART | Poor transfer | Amplitude mismatch |
| BLE | Poor transfer | Burst magnitude mismatch |
| ADC | Mixed | Parameter and input-definition mismatch |

---

## 3. System & Tools

- **Device:** ESP32-WROOM-32, 240 MHz
- **Firmware:** C
- **Current measurement:** Nordic PPK2, 100 kS/s
- **Transient measurement:** Oscilloscope with 1.7 Ω shunt
- **Analysis:** Python
- **Workloads:** CPU, UART, BLE, ADC
- **ADC setup:** ADC1 CH6 (GPIO34), 12-bit, 11 dB attenuation

---

## 4. Model

A first-order ODE was used to model the current response:

$$
\frac{dI(t)}{dt}
=
\frac{I_{\mathrm{idle}}+\Delta I\,u(t)-I(t)}{\tau}
$$

where:

- $I_{\mathrm{idle}}$: idle current
- $\Delta I$: active current increase
- $u(t)$: workload input
- $\tau$: transition time constant

The initial parameter set was identified from the CPU100 step-response workload:

- $I_{\mathrm{idle}} = 47.27\ \mathrm{mA}$
- $\Delta I = 20.19\ \mathrm{mA}$
- $\tau = 0.49\ \mathrm{ms}$

---

## 5. Validation Strategy

The validation followed four stages:

1. **CPU in-domain validation**  
   The CPU100-derived model was tested on unseen CPU workloads with different duty ratios and periods.

2. **Peripheral transferability**  
   The same CPU-derived model was applied to UART, BLE, and ADC workloads without changing the parameters.

3. **ADC re-identification**  
   The first-order ODE structure was kept, while the parameters were re-estimated using ADC-specific measurement data.

4. **Failure analysis**  
   Remaining errors were checked to determine whether they came from the parameters, the time constant, timing mismatch, or the workload input definition.

---

## 6. Key Results

**CPU**

The CPU100-derived model generalized well across different duty ratios and periods when the busy/wait timing was represented explicitly. Constant fractional inputs produced much larger errors.

**UART**

The measured TX-related current increase was close to the idle fluctuation level. The CPU-derived model therefore overpredicted UART activity because its current amplitude was too large.

**BLE**

BLE advertising produced clear interval-dependent current bursts. The CPU-derived model captured the interval trend but strongly underestimated the measured response magnitude.

**ADC**

The CPU-derived model showed different errors depending on the workload. Longer bursts were overpredicted, while Single 1 ms was strongly underpredicted, which motivated ADC-specific re-identification.

---

## 7. ADC Re-Identification

The CPU-derived parameters did not represent ADC current behavior well.

Therefore, the same first-order ODE was kept and the parameters were re-estimated using a continuous ADC workload.

- $I_{\mathrm{idle}} = 45.58\ \mathrm{mA}$
- $\Delta I = 12.94\ \mathrm{mA}$
- $\tau_{\mathrm{PPK2}} = 0.579\ \mathrm{ms}$
- $\tau_{\mathrm{scope}} \approx 0.44\ \mathrm{ms}$

The oscilloscope was used as an independent high-resolution check of the ADC transition.

During validation, the effect of the two $\tau$ estimates was compared while keeping $\Delta I$ fixed.

---

## 8. Key Failure Analysis

**Burst 100 / 1000**

After ADC re-identification, the predicted mean current was close to the measured value for the longer burst workloads.

However, the pointwise MAE remained relatively large.

Waveform analysis showed a repeatable timing offset of about 4.5–5 ms between measured and predicted burst transitions.

The additional drift over the 20 s active window was small, about 0.34–0.35 ms in total.

This showed that the remaining error was mainly caused by timing mismatch rather than amplitude mismatch.

**Single 1 ms**

The Single 1 ms workload showed a different failure.

The original model represented each ADC read as a 44 µs pulse every 1 ms, but the measured current stayed close to the continuous ADC active level.

To test the input definition, the model structure and ADC parameters were kept unchanged, and only the input was changed to continuous $u(t)=1$.

This reduced MAE from **12.84 mA to 0.58 mA**.

The result showed that the main failure came from the workload input definition rather than from a clear failure of the first-order ODE structure.

---

## 9. Main Takeaway

There was no single cause of prediction failure.

The CPU-derived parameters did not transfer well to peripheral workloads. After ADC-specific re-identification, the first-order structure remained useful, but the Single 1 ms workload still failed because the workload input did not represent the physical power state correctly.

Validation therefore requires checking why measured and predicted waveforms differ, not only how large the error is.

Through this project, I learned how to identify the practical boundary of a model and how to use discrepancies as clues to understand why a prediction fails.

---

## 10. Repository Structure

The repository is organized by measurement data, firmware, analysis scripts, and generated results.

```text
.
├── data/
│   ├── raw/
│   │   └── v3_ppk/
│   └── processed/
│       └── v3_ppk/
├── firmware/
│   └── v3_ppk/
│       ├── cpu_100/
│       ├── uart/
│       ├── ble/
│       └── adc/
├── scripts/
│   └── v3_ppk/
│       ├── peripheral/
│       └── core_only/
├── results/
│   └── v3_ppk/
│       └── peripheral/
└── README.md
```

- `data/raw` — original measurement data
- `data/processed` — processed datasets used for analysis
- `firmware` — ESP32 workload firmware
- `scripts` — Python analysis and validation scripts
- `results` — generated plots and numerical outputs

The `v3_ppk` directories contain the main datasets, firmware, analysis scripts, and results used in the final thesis workflow.

---

## 11. How to Reproduce

The analysis scripts are written in Python and use the measured datasets stored in the repository.

1. Clone the repository.
2. Install the required Python packages.
3. Run the analysis scripts for the target workload.
4. Generated plots and result files are saved under the `results/` directory.

### Environment

- Python 3.12.3
- NumPy
- pandas
- Matplotlib
- SciPy

### Example

Run the CPU time-constant estimation from the repository root:

```bash
python3 scripts/v3_ppk/estimate_tau_cpu_100.py
```