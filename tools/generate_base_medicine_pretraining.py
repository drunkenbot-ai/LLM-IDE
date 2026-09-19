"""Frontier Clinical Medicine, Pharmacology & Molecular Genetics Base Pretraining Generator.

Produces high-fidelity, textbook-grade clinical cases, pharmacological analyses,
and molecular genetics pretraining corpora strictly partitioned into cluster-ready
shards under 28.0 MB (29,360,128 bytes).

Domains Covered:
1. Clinical Case Presentations & Differential Diagnoses (USMLE Step 2/3 / Internal Medicine)
2. Pharmacology, Pharmacokinetics & Receptor Biophysics (PK/PD modeling, CYP450, Lineweaver-Burk)
3. Molecular Genetics, Genomics & Oncogenomics (CRISPR-Cas9, NGS Phred scores, DNA Repair)
4. Critical Care, Emergency Medicine & Hemodynamic Resuscitation (ARDS, Sepsis-3, Vasopressors)
5. Clinical Neurology, Stroke & Neurocritical Care (NIHSS, Thrombolysis, EVT, Status Epilepticus)
6. Nephrology, Acid-Base Physiology & Electrolyte Disorders (Anion Gap, Delta-Delta, Hyponatremia)
7. Infectious Disease & Antimicrobial Stewardship (ESBL, MRSA, Carbapenemases, HIV ART)
8. Endocrinology & Acute Metabolic Emergencies (DKA, HHS, Adrenal Crisis, Thyroid Storm)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

MAX_PARTITION_BYTES = 28 * 1024 * 1024  # 28 MB cluster node ceiling


class ShardedMedicineWriter:
    """Writes JSONL base pretraining records into partitioned files under 28 MB."""

    def __init__(self, output_dir: Path, prefix: str = "medicine_part", max_bytes: int = MAX_PARTITION_BYTES) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.part_idx = 1
        self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:03d}.jsonl"
        self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
        self.cur_bytes = 0
        self.cur_records = 0
        self.total_records = 0
        self.total_bytes = 0
        self.written_files: List[Path] = [self.cur_file]

    def write_record(self, text: str, subspecialty: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        doc = {
            "text": text,
            "meta": {
                "domain": "medicine",
                "subspecialty": subspecialty,
                "word_count": len(text.split()),
                "tokens": max(1, len(text) // 4),
                **(metadata or {})
            }
        }
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        b_len = len(line.encode("utf-8"))

        if self.cur_bytes + b_len > self.max_bytes and self.cur_records > 0:
            self.cur_fp.close()
            mb = self.cur_bytes / (1024 * 1024)
            print(f"  Saved {self.cur_file.name}: {self.cur_records:,} records ({mb:.2f} MB)")
            self.part_idx += 1
            self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:03d}.jsonl"
            self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
            self.written_files.append(self.cur_file)
            self.cur_bytes = 0
            self.cur_records = 0

        self.cur_fp.write(line)
        self.cur_bytes += b_len
        self.cur_records += 1
        self.total_records += 1
        self.total_bytes += b_len

    def close(self) -> None:
        if not self.cur_fp.closed:
            self.cur_fp.close()
            if self.cur_records > 0:
                mb = self.cur_bytes / (1024 * 1024)
                print(f"  Saved {self.cur_file.name}: {self.cur_records:,} records ({mb:.2f} MB)")
            elif len(self.written_files) > 1 and self.cur_file.exists() and self.cur_file.stat().st_size == 0:
                self.cur_file.unlink()
                self.written_files.remove(self.cur_file)


# ==============================================================================
# 1. Clinical Case Presentations & Differential Diagnosis (Internal Medicine)
# ==============================================================================

def generate_cardiology_case(rng: random.Random) -> str:
    age = rng.randint(48, 79)
    sex = rng.choice(["male", "female"])
    onset_hrs = rng.randint(1, 8)
    sbp = rng.randint(85, 175)
    dbp = rng.randint(52, 98)
    hr = rng.randint(62, 118)
    rr = rng.randint(18, 28)
    spo2 = rng.randint(91, 98)
    troponin = round(rng.uniform(1.8, 48.5), 2)
    bnp = rng.randint(450, 2800)
    cr = round(rng.uniform(0.9, 2.6), 2)
    culprit = rng.choice([
        ("anterior wall STEMI", "proximal left anterior descending (LAD) coronary artery", "V1-V4", "loss of anterior wall contractility with apical akinesis"),
        ("inferior wall STEMI", "proximal to mid right coronary artery (RCA)", "II, III, aVF", "inferior wall hypokinesis with preserved basal function"),
        ("lateral wall STEMI", "dominant left circumflex (LCx) obtuse marginal branch", "I, aVL, V5-V6", "lateral wall akinesis"),
    ])
    dapt_agent = rng.choice(["Ticagrelor 180 mg loading dose", "Prasugrel 60 mg loading dose"])

    template = r"""# Clinical Case Presentation: Acute Coronary Syndrome & Myocardial Infarction
**Specialty**: Cardiovascular Medicine / Interventional Cardiology
**Acuity**: Emergent | **Setting**: Cardiac Intensive Care Unit (CICU)

## 1. Patient History & Clinical Presentation
The patient is a __AGE__-year-old __SEX__ with a medical history of essential hypertension, type 2 diabetes mellitus (HbA1c 8.4%), and a 30 pack-year tobacco smoking history who presents to the Emergency Department with acute retrosternal chest pressure radiating to the left jaw and inner arm. The pain began __ONSET_HRS__ hours prior to arrival while at rest, described as crushing, 9/10 in severity, associated with diaphoresis, lightheadedness, and dyspnea. The patient has had no prior history of myocardial infarction or coronary revascularization.

### Physical Examination:
- **General**: Anxious, pale, diaphoretic, in moderate respiratory distress.
- **Vital Signs**: Blood Pressure: __SBP__/__DBP__ mmHg; Heart Rate: __HR__ beats/min (regular); Respiratory Rate: __RR__ breaths/min; SpO2: __SPO2__% on ambient air; Temperature: 37.1°C.
- **Cardiovascular**: S1 and S2 present; soft S4 gallop heard at the apex. No pericardial friction rub or holosystolic murmurs of acute mitral regurgitation or ventricular septal rupture. Jugular venous pressure estimated at 8 cm H2O.
- **Pulmonary**: Bilateral basilar crackles extending to the lower one-third of the lung fields, consistent with Killip Class II heart failure.
- **Abdomen**: Soft, non-tender, non-distended; no hepatomegaly.
- **Extremities**: Peripheral pulses symmetrical (2+ radial and dorsalis pedis); no significant peripheral pitting edema.

## 2. Diagnostic Workup & Laboratory Biomarkers
```
Hematology & Coagulation:
  White Blood Cell (WBC): 12.4 x 10^3/uL (neutrophil-predominant stress leukocytosis)
  Hemoglobin: 13.8 g/dL; Hematocrit: 41.2%
  Platelets: 242 x 10^3/uL
  Prothrombin Time (PT): 12.1 s | INR: 1.02 | aPTT: 28.4 s

Comprehensive Metabolic Panel:
  Serum Sodium: 138 mEq/L | Potassium: 4.1 mEq/L
  BUN: 22 mg/dL | Serum Creatinine: __CR__ mg/dL (eGFR: 48 mL/min/1.73m^2)
  Serum Glucose: 218 mg/dL | Serum Lactate: 2.1 mmol/L

Cardiac Biomarkers:
  High-Sensitivity Cardiac Troponin I (hs-cTnI): __TROPONIN__ ng/mL (Reference: < 0.04 ng/mL)
  B-type Natriuretic Peptide (BNP): __BNP__ pg/mL (Reference: < 100 pg/mL)
  Creatine Kinase-MB (CK-MB): 34.2 ng/mL | Relative Index: 6.8%
```

### 12-Lead Electrocardiogram (ECG):
Sinus rhythm at __HR__ bpm. Significant ST-segment elevation (> 2.5 mm in contiguous leads __LEADS__) with reciprocal ST-segment depression in opposite vector leads. Findings satisfy formal ACC/AHA criteria for acute __INFARCT_TYPE__.

### Transthoracic Echocardiogram (TTE):
Left ventricular ejection fraction (LVEF) estimated at 38%. Significant regional wall motion abnormality characterized by __ECHO_FINDING__. No severe mitral regurgitation, ventricular septal defect, or pericardial effusion detected.

## 3. Pathophysiology & Diagnostic Reasoning
The underlying mechanism is acute atherothrombotic plaque rupture (Type 1 Myocardial Infarction). Erosion or disruption of a fibrous cap overlying an atheroma exposes thrombogenic subendothelial collagen and tissue factor, triggering rapid platelet adhesion (via glycoprotein Ib/IX-V and von Willebrand factor), platelet activation with thromboxane A2 and ADP release, and subsequent glycoprotein IIb/IIIa cross-linking by fibrinogen. The propagation of a fibrin-rich thrombus leads to transmural coronary artery occlusion of the __CULPRIT__, causing myocardial ischemia, rapid ATP depletion, anaerobic glycolysis, and subsequent membrane disruption releasing intracellular troponin complexes into the circulation.

## 4. Evidence-Based Therapeutic Strategy
1. **Immediate Antiplatelet & Anticoagulant Therapy**:
   - Aspirin 324 mg chewable orally (irreversible COX-1 inhibition).
   - Dual Antiplatelet Therapy (DAPT): __DAPT_AGENT__ (P2Y12 receptor antagonist).
   - Intravenous unfractionated heparin (UFH) weight-based bolus (60 units/kg, max 4,000 units) followed by infusion (12 units/kg/hr) titrated to a therapeutic target aPTT of 50-70 seconds.
2. **Emergent Invasive Revascularization**:
   - Immediate transfer to the Cardiac Catheterization Laboratory for Primary Percutaneous Coronary Intervention (PPCI). Target door-to-balloon time < 90 minutes.
   - Right radial artery access approach favored to minimize major bleeding complications. Diagnostic angiography followed by aspiration thrombectomy and deployment of a second-generation drug-eluting stent (DES).
3. **Medical Optimization & Secondary Prevention**:
   - High-intensity statin therapy: Atorvastatin 80 mg daily (pleiotropic plaque stabilization).
   - Beta-blocker initiation: Metoprolol succinate 25 mg daily once hemodynamically stable without cardiogenic shock or active bronchospasm.
   - ACE-inhibitor initiation: Lisinopril 5 mg daily initiated within 24 hours to attenuate adverse left ventricular remodeling.
"""
    return (
        template
        .replace("__AGE__", str(age))
        .replace("__SEX__", sex)
        .replace("__ONSET_HRS__", str(onset_hrs))
        .replace("__SBP__", str(sbp))
        .replace("__DBP__", str(dbp))
        .replace("__HR__", str(hr))
        .replace("__RR__", str(rr))
        .replace("__SPO2__", str(spo2))
        .replace("__TROPONIN__", str(troponin))
        .replace("__BNP__", str(bnp))
        .replace("__CR__", str(cr))
        .replace("__INFARCT_TYPE__", culprit[0])
        .replace("__CULPRIT__", culprit[1])
        .replace("__LEADS__", culprit[2])
        .replace("__ECHO_FINDING__", culprit[3])
        .replace("__DAPT_AGENT__", dapt_agent)
    )


# ==============================================================================
# 2. Pharmacology, Pharmacokinetics & Receptor Biophysics
# ==============================================================================

def generate_pharmacokinetics_analysis(rng: random.Random) -> str:
    drug = rng.choice([
        ("Vancomycin", "Glycopeptide antibiotic", "Inhibition of cell wall synthesis by binding D-Ala-D-Ala", 0.7, 0.08, 6.0, "AUC/MIC ratio >= 400"),
        ("Digoxin", "Cardiac glycoside", "Reversible inhibition of Na+/K+-ATPase pump", 7.0, 0.05, 36.0, "Therapeutic trough concentration 0.5 - 0.9 ng/mL"),
        ("Theophylline", "Methylxanthine phosphodiesterase inhibitor", "Non-selective PDE inhibition & adenosine receptor antagonism", 0.5, 0.04, 8.0, "Therapeutic serum level 10 - 20 mcg/mL"),
        ("Gentamicin", "Aminoglycoside", "Binds 30S ribosomal subunit causing mRNA misreading", 0.25, 0.09, 2.5, "Peak concentration > 8x MIC; trough < 1.0 mcg/mL"),
    ])
    weight_kg = rng.randint(60, 95)
    clearance_l_hr = round(drug[4] * weight_kg * rng.uniform(0.7, 1.3), 2)
    vd_liters = round(drug[3] * weight_kg, 1)
    half_life_calc = round((0.693 * vd_liters) / clearance_l_hr, 1)
    cyp_target = rng.choice(["CYP3A4", "CYP2D6", "CYP2C19", "CYP2C9"])

    template = r"""# Quantitative Pharmacology & Pharmacokinetic/Pharmacodynamic (PK/PD) Analysis
**Agent**: __DRUG_NAME__ (__DRUG_CLASS__)
**Mechanism of Action**: __MOA__

## 1. Classical Two-Compartment Pharmacokinetic Model
Following intravenous administration, drug disposition exhibits multi-exponential decay described by a central compartment (blood and highly perfused vascular organs) and a peripheral tissue compartment:

$$\frac{dC_c(t)}{dt} = - (k_{10} + k_{12}) C_c(t) + k_{21} \frac{V_p}{V_c} C_p(t)$$

$$\frac{dC_p(t)}{dt} = k_{12} \frac{V_c}{V_p} C_c(t) - k_{21} C_p(t)$$

Where:
- $C_c(t), C_p(t)$: Drug concentration in central and peripheral compartments at time $t$.
- $V_c, V_p$: Apparent volume of central and peripheral compartments.
- $k_{12}, k_{21}$: Inter-compartmental distribution rate constants.
- $k_{10}$: First-order systemic elimination rate constant from the central compartment ($k_{10} = \frac{CL}{V_c}$).

The analytical solution for the central plasma concentration trajectory is:
$$C_c(t) = A e^{-\alpha t} + B e^{-\beta t}$$

Where $\alpha$ represents the distribution phase rate constant and $\beta$ represents the terminal elimination phase rate constant.

## 2. Patient-Specific Pharmacokinetic Parameters
```
Physical & Physiological Constants:
  Total Body Weight (TBW): __WEIGHT__ kg
  Apparent Volume of Distribution (Vd_beta): __VD__ L (__VD_PER_KG__ L/kg)
  Systemic Clearance (CL): __CLEARANCE__ L/hr
  Calculated Terminal Half-Life (t_1/2): __HALF_LIFE__ hours
  Target Exposure Metric: __TARGET_METRIC__
```

### Mathematical Derivation of Elimination Kinetics:
1. **Elimination Rate Constant ($k_e$)**:
   $$k_e = \frac{CL}{V_d} = \frac{__CLEARANCE__\text{ L/hr}}{__VD__\text{ L}} = \mathbf{__KE_VAL__\text{ hr}^{-1}}$$

2. **Terminal Elimination Half-Life ($t_{1/2}$)**:
   $$t_{1/2} = \frac{\ln(2)}{k_e} = \frac{0.693}{__KE_VAL__\text{ hr}^{-1}} = \mathbf{__HALF_LIFE__\text{ hours}}$$

3. **Steady-State Area Under the Curve ($AUC_{0-24, \tau}$)**:
   $$AUC_{0-24} = \frac{\text{Dose}_{24}}{CL}$$

## 3. Receptor Biophysics & Lineweaver-Burk Antagonism
Binding affinity to the target macromolecule follows the Hill-Langmuir isotherm:
$$\theta = \frac{[L]^n}{K_d^n + [L]^n}$$

Where:
- $\theta$: Fractional receptor occupancy ($[RL] / [R]_{total}$).
- $[L]$: Free unbound ligand concentration.
- $K_d$: Equilibrium dissociation constant ($K_d = k_{off} / k_{on}$).
- $n$: Hill cooperativity coefficient ($n > 1$ denotes positive cooperativity).

In the presence of a reversible competitive inhibitor, the apparent Michaelis constant $K_m$ increases while the maximal velocity $V_{max}$ remains unaltered:
$$\frac{1}{v} = \frac{K_m}{V_{max}} \left(1 + \frac{[I]}{K_i}\right) \frac{1}{[S]} + \frac{1}{V_{max}}$$

## 4. Cytochrome P450 Metabolism & Drug-Drug Interactions (DDI)
Primary hepatic biotransformation involves oxidative phase I metabolism mediated predominantly by __CYP_ENZYME__. 
- **Co-administration with Potent Inducers** (e.g., Rifampin, Carbamazepine, St. John's Wort) accelerates phase I clearance, depressing steady-state systemic exposure ($AUC_{ss}$) by up to 60-80%, resulting in therapeutic failure.
- **Co-administration with Potent Inhibitors** (e.g., Ketoconazole, Clarithromycin, Ritonavir) competitively binds the catalytic heme iron center of __CYP_ENZYME__, shifting the elimination curve and precipitating drug toxicity. Therapeutic drug monitoring (TDM) is indicated to maintain concentrations within the validated therapeutic window.
"""
    ke_val = round(clearance_l_hr / vd_liters, 4)
    vd_per_kg = round(vd_liters / weight_kg, 2)
    return (
        template
        .replace("__DRUG_NAME__", drug[0])
        .replace("__DRUG_CLASS__", drug[1])
        .replace("__MOA__", drug[2])
        .replace("__WEIGHT__", str(weight_kg))
        .replace("__VD__", str(vd_liters))
        .replace("__VD_PER_KG__", str(vd_per_kg))
        .replace("__CLEARANCE__", str(clearance_l_hr))
        .replace("__HALF_LIFE__", str(half_life_calc))
        .replace("__TARGET_METRIC__", drug[6])
        .replace("__KE_VAL__", str(ke_val))
        .replace("__CYP_ENZYME__", cyp_target)
    )


# ==============================================================================
# 3. Molecular Genetics, Genomics & Oncogenomics
# ==============================================================================

def generate_molecular_genetics_analysis(rng: random.Random) -> str:
    topic = rng.choice([
        (
            "CRISPR-Cas9 Precise Gene Editing & DNA Repair Pathways",
            "Streptococcus pyogenes Cas9 (SpCas9)",
            "5'-NGG-3'",
            "Non-Homologous End Joining (NHEJ)",
            "Homology-Directed Repair (HDR)",
            "frameshift indels causing premature stop codon and nonsense-mediated mRNA decay",
            "single-stranded donor oligonucleotide (ssODN) containing engineered homologous arm sequences"
        ),
        (
            "Next-Generation Sequencing (NGS) Variant Calling & Quality Metrics",
            "Illumina Sequencing-by-Synthesis (SBS)",
            "Flow cell reversible terminator fluorophores",
            "Phred Quality Score Filtering",
            "Bayesian Haplotype Assembly",
            "misincorporation error rate rising toward the 3' terminus of paired-end reads",
            "GATK HaplotypeCaller emitting multi-allelic candidate indels and single nucleotide variants (SNVs)"
        ),
        (
            "Oncogenic Driver Mutations & Targeted Signal Transduction",
            "Receptor Tyrosine Kinase / MAPK Pathway",
            "BRAF V600E (Valine to Glutamic Acid substitution at codon 600)",
            "Constitutive RAF monomer kinase domain activation",
            "Dual inhibition via BRAF inhibitor (Dabrafenib) + MEK inhibitor (Trametinib)",
            "hyperphosphorylation of downstream ERK1/2 promoting uncontrolled cell cycle progression",
            "overcoming paradoxical MAPK activation in BRAF wild-type RAS-mutant clones"
        )
    ])
    gc_content = rng.randint(42, 64)
    phred_q = rng.randint(30, 42)
    error_prob = round(10 ** (-phred_q / 10), 6)

    template = r"""# Molecular Genetics & Genomic Biotechnology: __TITLE__
**Discipline**: Molecular Biology, Oncogenomics & Genomic Engineering

## 1. Biophysical & Enzymatic Mechanism
The molecular architecture of __ENZYME_OR_PLATFORM__ operates through coordinated sequence recognition and catalytic cleavage.
1. **Target Specificity & Hybridization**:
   - The synthetic single-guide RNA (sgRNA) comprises a 20-nucleotide crRNA spacer sequence coupled to a trans-activating crRNA (tracrRNA) scaffold.
   - Target recognition requires complementary base pairing adjacent to a strict Protospacer Adjacent Motif (PAM): `__PAM_OR_FEATURE__`.
   - Structural unzipping of the genomic double helix forms a stable R-loop configuration.

2. **Catalytic Cleavage**:
   - Endonuclease cleavage is executed by dual catalytic domains:
     - The **RuvC domain** cleaves the non-target DNA strand.
     - The **HNH domain** cleaves the target DNA strand complementary to the guide.
   - Cleavage occurs precisely 3 base pairs upstream of the PAM, generating a blunt-ended double-strand break (DSB).

## 2. Competitive DNA Repair Kinetics
Following double-strand break induction, mammalian cell survival depends on two competing enzymatic repair pathways:

```
Double-Strand Break (DSB)
          │
          ├──> Pathway A: __PATHWAY_A__ (Error-Prone, Default)
          │    Kinetics: Fast (t_1/2 ~ 30 min)
          │    Mechanism: Ku70/Ku80 heterodimer binding -> DNA-PKcs recruitment ->
          │               Artemis end-processing -> DNA Ligase IV / XRCC4 ligation.
          │    Outcome: __OUTCOME_A__.
          │
          └──> Pathway B: __PATHWAY_B__ (High-Fidelity, Cell-Cycle Dependent)
               Kinetics: Slow (t_1/2 ~ 4-8 hrs), active strictly in late S / G2 phase.
               Mechanism: MRN complex (Mre11-Rad50-Nbs1) + CtIP 5'-to-3' resection ->
                          RPA single-strand coating -> Rad51 filament nucleoprotein assembly ->
                          Template strand invasion and displacement loop (D-loop) synthesis.
               Outcome: __OUTCOME_B__.
```

## 3. High-Throughput Quantitative Sequencing Metrics
High-confidence variant detection requires statistical modeling of sequencing error rates using the standard Phred quality scoring system:

$$Q = -10 \log_{10}(P_{\text{error}})$$

$$P_{\text{error}} = 10^{-\frac{Q}{10}}$$

For a base position with a Phred score of $Q = \mathbf{__PHRED__}$:
- Base call accuracy: **__ACCURACY__%**
- Probability of incorrect base call: $P_{\text{error}} = 10^{-\frac{__PHRED__}{10}} = \mathbf{__ERROR_PROB__}$
- Local genomic region GC content: **__GC_CONTENT__%**

### Variant Allele Frequency (VAF) & Tumor Heterogeneity:
$$VAF = \frac{N_{\text{alt}}}{N_{\text{ref}} + N_{\text{alt}}}$$

Where $N_{\text{alt}}$ represents mutant read depth and $N_{\text{ref}}$ represents wild-type read depth. In somatic oncology testing, subclonal driver mutations often display $VAF \in [0.05, 0.25]$, requiring high sequencing depth ($\ge 500\times$) to differentiate true biological variants from PCR amplification artifacts and deamination background noise.
"""
    accuracy = round((1.0 - error_prob) * 100, 4)
    return (
        template
        .replace("__TITLE__", topic[0])
        .replace("__ENZYME_OR_PLATFORM__", topic[1])
        .replace("__PAM_OR_FEATURE__", topic[2])
        .replace("__PATHWAY_A__", topic[3])
        .replace("__PATHWAY_B__", topic[4])
        .replace("__OUTCOME_A__", topic[5])
        .replace("__OUTCOME_B__", topic[6])
        .replace("__PHRED__", str(phred_q))
        .replace("__ERROR_PROB__", f"{error_prob:.6f}")
        .replace("__ACCURACY__", str(accuracy))
        .replace("__GC_CONTENT__", str(gc_content))
    )


# ==============================================================================
# 4. Critical Care, Emergency Medicine & Hemodynamics
# ==============================================================================

def generate_critical_care_analysis(rng: random.Random) -> str:
    shock_type = rng.choice([
        (
            "Distributive Shock (Septic Shock / Sepsis-3)",
            "Systemic Vasodilation & Endothelial Microvascular Leak",
            "Low SVR (< 700 dynes*s/cm^5), High Cardiac Output (> 8.0 L/min initially), Low MAP",
            "Norepinephrine (alpha-1 > beta-1 agonist)",
            "Vasopressin (V1 receptor agonist at fixed 0.03 units/min)",
            "30 mL/kg IV crystalloid (Balanced Salt Solution) within 3 hours + broad-spectrum IV antimicrobials within 1 hour"
        ),
        (
            "Cardiogenic Shock (Post-Infarction Pump Failure)",
            "Primary Myocardial Inotropic Failure & Elevated Filling Pressures",
            "High SVR (> 1400 dynes*s/cm^5), Low Cardiac Index (< 2.2 L/min/m^2), Elevated PCWP (> 18 mmHg)",
            "Dobutamine (beta-1 inotrope)",
            "Inhaled Nitric Oxide / Mechanical Circulatory Support (Impella CP)",
            "Coronary revascularization, cautious diuresis, avoiding fluid boluses that worsen pulmonary capillary wedge pressure"
        ),
        (
            "Severe Acute Respiratory Distress Syndrome (ARDS)",
            "Diffuse Alveolar Capillary Membrane Disruption & Non-Cardiogenic Edema",
            "PaO2/FiO2 ratio <= 100 mmHg, Bilateral opacities on chest radiography, Low respiratory compliance (< 25 mL/cmH2O)",
            "Neuromuscular blockade (Cisatracurium infusion for 48h)",
            "Inhaled Epoprostenol / Prone Positioning Therapy (16 hrs/day)",
            "Low tidal volume lung-protective mechanical ventilation (6 mL/kg predicted body weight, plateau pressure < 30 cmH2O, titrated PEEP)"
        )
    ])
    map_target = rng.randint(65, 75)
    lactate = round(rng.uniform(2.4, 9.8), 1)
    co_val = round(rng.uniform(3.2, 8.5), 1)
    cvp_val = rng.randint(4, 16)
    sbp_val = rng.randint(70, 95)
    dbp_val = rng.randint(40, 58)
    calc_map = round(dbp_val + (sbp_val - dbp_val) / 3.0, 1)

    template = r"""# Intensive Care Medicine: Clinical Management of __SHOCK_TYPE__
**Specialty**: Critical Care Medicine / Surgical & Medical ICU
**Clinical Guideline**: Evidence-Based Resuscitation Protocols

## 1. Hemodynamic Physiology & Governing Equations
Hemodynamic monitoring requires continuous integration of pressure, flow, and vascular resistance parameters:

### 1. Mean Arterial Pressure (MAP):
$$MAP = DBP + \frac{1}{3}(SBP - DBP)$$

For a patient with SBP = __SBP__ mmHg and DBP = __DBP__ mmHg:
$$MAP = __DBP__ + \frac{1}{3}(__SBP__ - __DBP__) = \mathbf{__CALC_MAP__\text{ mmHg}}$$
*(Target MAP $\ge$ __MAP_TARGET__ mmHg to maintain renal and mesenteric perfusion).*

### 2. Systemic Vascular Resistance (SVR):
$$SVR = 80 \times \frac{MAP - CVP}{CO}$$

Where:
- $MAP$: Mean Arterial Pressure (__CALC_MAP__ mmHg).
- $CVP$: Central Venous Pressure (__CVP__ mmHg).
- $CO$: Cardiac Output (__CO__ L/min).

### 3. Arterial Oxygen Content ($CaO_2$) & Systemic Delivery ($DO_2$):
$$CaO_2 = (1.34 \times Hb \times SaO_2) + (0.0031 \times PaO_2)$$
$$DO_2 = CO \times CaO_2 \times 10$$

Oxygen consumption ($VO_2$) and the oxygen extraction ratio ($O_2ER = VO_2 / DO_2$) determine the adequacy of microcirculatory perfusion. When $DO_2$ drops below the critical threshold ($DO_{2,crit}$), cellular metabolism converts to anaerobic glycolysis, elevating serum lactate (__LACTATE__ mmol/L).

## 2. Pathophysiological Profile
- **Primary Etiology**: __ETIOLOGY__
- **Hemodynamic Fingerprint**: __FINGERPRINT__
- **Cellular Marker**: Serum Lactate = __LACTATE__ mmol/L (tissue hypoperfusion marker).

## 3. Protocolized Resuscitation Bundle
```
Step 1: Volumetric Resuscitation & Fluid Responsiveness
  __FLUID_STRATEGY__
  - Assess volume responsiveness dynamically using Pulse Pressure Variation (PPV > 12%)
    or passive leg raise (PLR) inducing a > 10% increase in stroke volume index (SVI).

Step 2: Vasoactive Agent Titration
  - Primary Vasopressor: __VASO_1__
    Target: MAP >= __MAP_TARGET__ mmHg.
  - Secondary Adjunct: __VASO_2__
    Initiate when primary vasopressor requirements escalate to mitigate high-dose catecholamine adrenergic toxicity.

Step 3: Advanced Hemodynamic & Metabolic Monitoring
  - Central venous catheter placement with continuous ScvO2 monitoring (target >= 70%).
  - Arterial line placement for beat-to-beat blood pressure measurement and serial arterial blood gases.
  - Serial lactate clearance tracking: aim for >= 20% clearance every 2 hours as a prognostic marker of cellular recovery.
```
"""
    return (
        template
        .replace("__SHOCK_TYPE__", shock_type[0])
        .replace("__ETIOLOGY__", shock_type[1])
        .replace("__FINGERPRINT__", shock_type[2])
        .replace("__VASO_1__", shock_type[3])
        .replace("__VASO_2__", shock_type[4])
        .replace("__FLUID_STRATEGY__", shock_type[5])
        .replace("__MAP_TARGET__", str(map_target))
        .replace("__LACTATE__", str(lactate))
        .replace("__CO__", str(co_val))
        .replace("__CVP__", str(cvp_val))
        .replace("__SBP__", str(sbp_val))
        .replace("__DBP__", str(dbp_val))
        .replace("__CALC_MAP__", str(calc_map))
    )


# ==============================================================================
# 5. Clinical Neurology, Stroke & Neurocritical Care
# ==============================================================================

def generate_neurology_analysis(rng: random.Random) -> str:
    nihss = rng.randint(8, 22)
    aspects = rng.randint(6, 10)
    onset_hrs = round(rng.uniform(1.2, 5.8), 1)
    vessel = rng.choice([
        ("M1 segment of the Left Middle Cerebral Artery (MCA)", "Right hemiparesis, right facial droop, expressive (Broca) and receptive (Wernicke) global aphasia"),
        ("M1 segment of the Right Middle Cerebral Artery (MCA)", "Left hemiplegia, left-sided hemineglect / spatial inattention, left homonymous hemianopia"),
        ("Basilar Artery", "Bilateral quadriparesis, dysarthria, horizontal gaze palsy, altered mental status with preserved vertical gaze"),
    ])
    thrombolysis_eligible = onset_hrs <= 4.5

    template = r"""# Neurocritical Care: Acute Ischemic Stroke & Revascularization Protocol
**Subspecialty**: Vascular Neurology & Neurointerventional Surgery
**Acuity**: Hyperacute Emergency

## 1. Acute Neurovascular Presentation & Assessment
A patient presents with acute onset focal neurological deficits starting __ONSET_HRS__ hours prior to arrival. 
- **Target Occlusion Site**: __VESSEL__
- **Clinical Semiology**: __SEMIOLOGY__
- **National Institutes of Health Stroke Scale (NIHSS)**: **__NIHSS__** (Severe Neurological Impairment).
- **Blood Pressure on Arrival**: 168/94 mmHg (Permissive hypertension target < 185/110 mmHg if candidate for thrombolysis).

## 2. Emergency Neuroimaging Protocol
```
Non-Contrast Head CT (NCCT):
  Alberta Stroke Program Early CT Score (ASPECTS): __ASPECTS__ / 10
  - Subcortical and cortical gray-white differentiation preserved across >= 7 vascular territories.
  - Hyperdense vessel sign identified in the target arterial bifurcation.
  - Exclusion of acute intracranial hemorrhage (ICH) and established mature hypodense territorial infarction (> 1/3 MCA territory).

CT Angiography (CTA) - Head and Neck:
  - Demonstrates abrupt abrupt contrast cutoff in the __VESSEL__.
  - Collateral circulation graded via Tan collateral scoring system as intermediate-to-robust.

CT Perfusion (CTP):
  - Ischemic Core (CBF < 30% normal brain tissue): Estimated at 18 mL.
  - Critical Penumbra (Tmax > 6.0 seconds delay): Estimated at 84 mL.
  - Mismatch Volume: 66 mL | Mismatch Ratio: 4.67 (Demonstrates substantial salvageable tissue).
```

## 3. Evidence-Based Revascularization Algorithm
1. **Intravenous Thrombolysis**:
   __THROMBOLYSIS_DECISION__

2. **Mechanical Thrombectomy (Endovascular Therapy - EVT)**:
   - **Indication**: Large Vessel Occlusion (LVO) in the anterior circulation within 24 hours of onset, meeting DEFUSE-3 / DAWN trial physiological inclusion criteria (penumbra-to-core mismatch on advanced imaging with baseline functional independence mRS 0-1).
   - **Approach**: Emergent transfemoral or transradial catheterization. Deployment of a stent-retriever system (e.g., Solitaire / Trevo) paired with distal large-bore aspiration catheter (ADAPT technique).
   - **Target Angiographic Result**: Modified Treatment in Cerebral Ischemia (mTICI) Grade 2b/3 revascularization (complete perfusion of the distal vascular tree).

3. **Post-Intervention Neurocritical Care ICU Protocol**:
   - Continuous arterial line blood pressure monitoring: strict BP target < 180/105 mmHg (if thrombolysis administered) or < 140/90 mmHg post-successful recanalization to minimize reperfusion injury and hemorrhagic transformation.
   - Serial neurological checks (NIHSS every 15 min for 2 hours, then every 30 min for 6 hours).
   - Repeat non-contrast CT at 24 hours prior to initiation of antiplatelet or anticoagulant therapy.
"""
    if thrombolysis_eligible:
        thrombo_text = (
            f"- **ELIGIBLE** (Onset {onset_hrs}h <= 4.5h window).\n"
            f"   - Agent: Tenecteplase (TNK-tPA) 0.25 mg/kg IV single bolus (max 25 mg) or Alteplase 0.9 mg/kg (10% bolus, 90% over 60 min).\n"
            f"   - Proceed to IV administration immediately without delaying transport to the Cath Lab for endovascular therapy."
        )
    else:
        thrombo_text = (
            f"- **INELIGIBLE FOR IV THROMBOLYSIS** (Onset {onset_hrs}h exceeds 4.5h standard therapeutic window).\n"
            f"   - Immediate priority: Proceed directly to catheterization suite for mechanical thrombectomy under extended window criteria."
        )

    return (
        template
        .replace("__ONSET_HRS__", str(onset_hrs))
        .replace("__VESSEL__", vessel[0])
        .replace("__SEMIOLOGY__", vessel[1])
        .replace("__NIHSS__", str(nihss))
        .replace("__ASPECTS__", str(aspects))
        .replace("__THROMBOLYSIS_DECISION__", thrombo_text)
    )


# ==============================================================================
# 6. Nephrology, Acid-Base Physiology & Electrolyte Disorders
# ==============================================================================

def generate_nephrology_acid_base_analysis(rng: random.Random) -> str:
    na = rng.randint(132, 146)
    cl = rng.randint(92, 108)
    hco3 = rng.randint(8, 22)
    ag = na - (cl + hco3)
    pco2 = rng.randint(18, 44)
    ph = round(6.1 + math.log10(hco3 / (0.03 * pco2)), 2)

    # Winter's formula calculation
    winters_expected_pco2 = round(1.5 * hco3 + 8, 1)
    delta_ag = ag - 12
    delta_hco3 = 24 - hco3
    delta_ratio = round(delta_ag / max(1, delta_hco3), 2)

    template = r"""# Quantitative Nephrology: Acid-Base Physiology & Arterial Blood Gas Analysis
**Subspecialty**: Renal Medicine & Clinical Acid-Base Physiology

## 1. Serum Electrolyte Panel & Calculated Anion Gap
```
Serum Chemistries:
  Sodium ([Na+]): __NA__ mEq/L
  Chloride ([Cl-]): __CL__ mEq/L
  Bicarbonate ([HCO3-]): __HCO3__ mEq/L
  Potassium ([K+]): 4.6 mEq/L
  Blood Urea Nitrogen (BUN): 38 mg/dL
  Serum Creatinine: 2.4 mg/dL (Baseline 0.9 mg/dL -> KDIGO Stage 2 Acute Kidney Injury)
```

### Mathematical Calculation of the Serum Anion Gap (AG):
$$AG = [\text{Na}^+] - ([\text{Cl}^-] + [\text{HCO}_3^-])$$

$$AG = __NA__ - (__CL__ + __HCO3__) = \mathbf{__AG__\text{ mEq/L}}$$
*(Normal Reference Range: $8 - 12\text{ mEq/L}$).*

## 2. Arterial Blood Gas (ABG) & Henderson-Hasselbalch Equilibrium
$$\text{pH} = 6.10 + \log_{10}\left(\frac{[\text{HCO}_3^-]}{0.0307 \times PaCO_2}\right)$$

Measured Arterial Values:
- **pH**: **__PH__** (Severe Acidemia)
- **$PaCO_2$**: **__PCO2__ mmHg**
- **$PaO_2$**: 94 mmHg on room air

### Assessment of Respiratory Compensation (Winter's Formula):
For primary metabolic acidosis, appropriate respiratory compensation by hyperventilation is determined by Winter's Formula:
$$Expected\ PaCO_2 = (1.5 \times [\text{HCO}_3^-]) + 8 \pm 2$$

$$Expected\ PaCO_2 = (1.5 \times __HCO3__) + 8 = \mathbf{__WINTERS__ \pm 2\text{ mmHg}}$$

- **Measured $PaCO_2$**: __PCO2__ mmHg
- **Physiological Interpretation**: __COMPENSATION_INTERP__

## 3. Delta-Delta ($\Delta - \Delta$) Ratio Analysis
In high anion gap metabolic acidosis, the Delta-Delta ratio evaluates for concurrent occult metabolic acid-base disturbances:

$$\Delta\Delta = \frac{\Delta\text{AG}}{\Delta[\text{HCO}_3^-]} = \frac{AG - 12}{24 - [\text{HCO}_3^-]}$$

$$\Delta\Delta = \frac{__AG__ - 12}{24 - __HCO3__} = \frac{__DELTA_AG__}{__DELTA_HCO3__} = \mathbf{__DELTA_RATIO__}$$

### Diagnostic Criteria:
- $\Delta\Delta < 0.8$: Mixed High Anion Gap Metabolic Acidosis **and** Concurrent Normal Anion Gap (Hyperchloremic) Metabolic Acidosis (e.g., diarrhea, renal tubular acidosis).
- $0.8 \le \Delta\Delta \le 2.0$: Pure High Anion Gap Metabolic Acidosis (e.g., DKA, Lactic Acidosis, Toxic Ingestions).
- $\Delta\Delta > 2.0$: Mixed High Anion Gap Metabolic Acidosis **and** Concurrent Metabolic Alkalosis (e.g., severe vomiting, concurrent diuretic use).

**Patient Classification**: __DELTA_INTERP__
"""
    if abs(pco2 - winters_expected_pco2) <= 2:
        comp_interp = "Appropriate respiratory compensation. Pure primary metabolic acidosis with physiological compensatory hyperventilation."
    elif pco2 > winters_expected_pco2 + 2:
        comp_interp = f"Measured PaCO2 ({pco2} mmHg) exceeds expected compensation ({winters_expected_pco2} mmHg). Demonstrates concurrent primary respiratory acidosis (hypoventilation / central respiratory depression)."
    else:
        comp_interp = f"Measured PaCO2 ({pco2} mmHg) is lower than expected compensation ({winters_expected_pco2} mmHg). Demonstrates concurrent primary respiratory alkalosis."

    if delta_ratio < 0.8:
        delta_interp = "Delta ratio < 0.8: Demonstrates combined High Anion Gap Metabolic Acidosis with concomitant Normal Anion Gap (Hyperchloremic) Acidosis."
    elif delta_ratio <= 2.0:
        delta_interp = "Delta ratio between 0.8 and 2.0: Consistent with pure High Anion Gap Metabolic Acidosis without secondary metabolic disturbances."
    else:
        delta_interp = "Delta ratio > 2.0: Demonstrates High Anion Gap Acidosis with a co-existing secondary Metabolic Alkalosis."

    return (
        template
        .replace("__NA__", str(na))
        .replace("__CL__", str(cl))
        .replace("__HCO3__", str(hco3))
        .replace("__AG__", str(ag))
        .replace("__PH__", str(ph))
        .replace("__PCO2__", str(pco2))
        .replace("__WINTERS__", str(winters_expected_pco2))
        .replace("__DELTA_AG__", str(delta_ag))
        .replace("__DELTA_HCO3__", str(delta_hco3))
        .replace("__DELTA_RATIO__", str(delta_ratio))
        .replace("__COMPENSATION_INTERP__", comp_interp)
        .replace("__DELTA_INTERP__", delta_interp)
    )


# ==============================================================================
# 7. Infectious Disease & Antimicrobial Stewardship
# ==============================================================================

def generate_infectious_disease_analysis(rng: random.Random) -> str:
    pathogen = rng.choice([
        ("Pseudomonas aeruginosa", "Gram-negative bacillus", "Loss of OprD porin + over-expression of AmpC cephalosporinase & MexAB-OprM efflux pump", "Cefepime or Piperacillin-Tazobactam + Tobramycin or Ciprofloxacin", "MDR / DTR (Difficult-to-Treat Resistance)"),
        ("Methicillin-Resistant Staphylococcus aureus (MRSA)", "Gram-positive cocci in clusters", "Acquisition of mecA gene on SCCmec element encoding penicillin-binding protein PBP2a with low affinity for beta-lactams", "Vancomycin (target AUC/MIC 400-600) or Daptomycin (8-10 mg/kg)", "Vancomycin intermediate resistance (VISA) through cell wall thickening"),
        ("Klebsiella pneumoniae (KPC-producing CRE)", "Gram-negative bacillus (Enterobacterales)", "Class A Klebsiella pneumoniae carbapenemase (KPC) hydrolyzing all beta-lactams and carbapenems", "Ceftazidime-avibactam or Meropenem-vaborbactam", "Extensive drug resistance (XDR)"),
    ])
    temp_c = round(rng.uniform(38.4, 40.1), 1)
    wbc = round(rng.uniform(14.5, 28.2), 1)
    bands = rng.randint(8, 22)
    crp = round(rng.uniform(85, 260), 1)

    template = r"""# Clinical Infectious Diseases: Antimicrobial Resistance Mechanisms & Targeted Stewardship
**Pathogen**: __PATHOGEN__ (__GRAM_STAIN__)
**Resistance Phenotype**: __RESISTANCE_CLASS__

## 1. Clinical Presentation & Microbiological Profile
A hospitalized patient develops acute septic shock on hospital day 7 with new-onset high fevers (Temperature: __TEMP__°C), chills, and hypotension.
- **Inflammatory Biomarkers**:
  - Peripheral White Blood Cell Count: **__WBC__ x 10^3/uL** with **__BANDS__% band forms** (severe left shift).
  - Serum C-Reactive Protein (CRP): **__CRP__ mg/L** (Normal: < 5.0 mg/L).
  - Serum Procalcitonin: 6.8 ng/mL (High positive predictive value for invasive bacterial bacteremia).
- **Blood & Sputum Cultures**: Preliminary Gram stain reveals **__GRAM_STAIN__**. Molecular rapid diagnostic panel (multiplex PCR) identifies genetic resistance determinants.

## 2. Molecular Resistance Mechanism
The molecular basis of treatment failure in this strain arises from:
$$\mathbf{__MECHANISM__}$$

### Biochemical Enzymatic Kinetics:
Hydrolysis of beta-lactam rings by serine-carbapenemases or beta-lactamases follows acylation and deacylation stages:
$$E + S \underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}} E \cdot S \xrightarrow{k_2} E\text{-}Acyl \xrightarrow{k_3} E + P$$

Where:
- $k_2$: Acylation rate constant forming the covalent acyl-enzyme intermediate.
- $k_3$: Deacylation (hydrolytic water molecule attack) freeing active enzyme $E$.
- In resistant strains, $k_3$ is extremely rapid ($\sim 1000\text{ s}^{-1}$), rapidly degrading penicillins, cephalosporins, and carbapenems before target transpeptidase binding occurs.

## 3. Evidence-Based Antimicrobial Regimen
```
Empiric & Targeted Therapeutic Coverage:
  First-Line Regimen: __TREATMENT__
  - Optimization Strategy: Prolonged or continuous intravenous infusions (e.g., extended infusion over 3-4 hours)
    to maximize the pharmacodynamic parameter Time above Minimal Inhibitory Concentration (%T > MIC >= 60-70%).
  
Stewardship & De-escalation Protocol:
  - Source Control: Removal of indwelling central venous access lines, drainage of loculated collections.
  - Repeat blood cultures every 48 hours until documented microbiological clearance.
  - De-escalation: Narrow antimicrobial spectrum upon receipt of automated microdilution phenotypic AST MICs.
```
"""
    return (
        template
        .replace("__PATHOGEN__", pathogen[0])
        .replace("__GRAM_STAIN__", pathogen[1])
        .replace("__MECHANISM__", pathogen[2])
        .replace("__TREATMENT__", pathogen[3])
        .replace("__RESISTANCE_CLASS__", pathogen[4])
        .replace("__TEMP__", str(temp_c))
        .replace("__WBC__", str(wbc))
        .replace("__BANDS__", str(bands))
        .replace("__CRP__", str(crp))
    )


# ==============================================================================
# 8. Endocrinology & Acute Metabolic Emergencies
# ==============================================================================

def generate_endocrinology_analysis(rng: random.Random) -> str:
    glucose = rng.randint(380, 920)
    ketones = round(rng.uniform(3.8, 9.4), 1)
    k_level = round(rng.uniform(3.2, 5.8), 1)
    na_level = rng.randint(126, 136)
    corrected_na = round(na_level + 0.016 * (glucose - 100), 1)
    serum_osm = round(2 * na_level + glucose / 18 + 24 / 2.8, 1)

    template = r"""# Clinical Endocrinology: Management of Severe Diabetic Ketoacidosis (DKA)
**Specialty**: Endocrinology, Diabetes & Acute Metabolism
**Setting**: Medical Intensive Care Unit (MICU)

## 1. Pathophysiology & Hormonal Imbalance
Diabetic Ketoacidosis represents a state of absolute or profound relative insulin deficiency coupled with counter-regulatory hormone excess (glucagon, catecholamines, cortisol, and growth hormone).

$$\text{Insulin} \downarrow \quad \times \quad \text{Glucagon, Epinephrine} \uparrow$$

1. **Hepatic Gluconeogenesis & Glycogenolysis**: Unchecked activation of phosphoenolpyruvate carboxykinase (PEPCK) and fructose 1,6-bisphosphatase stimulates massive hepatic glucose output, driving profound hyperglycemia (__GLUCOSE__ mg/dL) and glycosuria-induced osmotic diuresis.
2. **Adipose Lipolysis & Ketogenesis**: Insulin deficiency disinhibits hormone-sensitive lipase (HSL), releasing free fatty acids (FFAs) into the portal circulation. Unopposed carnitine palmitoyltransferase-1 (CPT-1) shuttles acyl-CoA into mitochondrial matrix for beta-oxidation, yielding excess acetyl-CoA that condenses into acetoacetate and beta-hydroxybutyrate (__KETONES__ mmol/L).

## 2. Emergency Laboratory Diagnostic Panel
```
Metabolic Profile:
  Serum Glucose: __GLUCOSE__ mg/dL
  Serum Beta-Hydroxybutyrate: __KETONES__ mmol/L (Diagnostic threshold > 3.0 mmol/L)
  Serum Sodium ([Na+]): __NA__ mEq/L
  Serum Potassium ([K+]): __K_LEVEL__ mEq/L
  Arterial Blood Gas pH: 7.12 | Bicarbonate ([HCO3-]): 10 mEq/L
  Calculated Serum Anion Gap: 24 mEq/L (High Anion Gap Acidosis)
```

### Osmotic & Pseudohyponatremic Correction:
1. **Corrected Sodium (Katz formula)**:
   $$[\text{Na}^+]_{\text{corrected}} = [\text{Na}^+] + 0.016 \times (\text{Glucose} - 100)$$
   $$[\text{Na}^+]_{\text{corrected}} = __NA__ + 0.016 \times (__GLUCOSE__ - 100) = \mathbf{__CORRECTED_NA__\text{ mEq/L}}$$

2. **Effective Serum Osmolality**:
   $$\text{Osm}_{\text{eff}} = 2[\text{Na}^+] + \frac{\text{Glucose}}{18} = 2(__NA__) + \frac{__GLUCOSE__}{18} = \mathbf{__SERUM_OSM__\text{ mOsm/kg}}$$

## 3. Protocolized Resuscitation & Insulin Protocol
```
Phase 1: Volume Resuscitation
  - Initial 1-2 hours: 1,000 - 1,500 mL/hr of 0.9% Normal Saline or Balanced Crystalloid to restore intravascular volume.
  - Subsequent hours: 250 - 500 mL/hr. Switch to 0.45% NaCl if corrected sodium is normal or elevated.
  - When serum glucose drops <= 250 mg/dL: Add 5% Dextrose (D5W 0.45% NaCl) to prevent rapid osmolar collapse and cerebral edema.

Phase 2: Potassium Repletion Strategy
  - Potassium level = __K_LEVEL__ mEq/L:
    * If K+ < 3.3 mEq/L: HOLD insulin; administer 20-40 mEq/hr IV potassium until K+ > 3.3 mEq/L.
    * If K+ 3.3 - 5.2 mEq/L: Administer 20-30 mEq K+ per liter of IV fluid; maintain serum K+ between 4.0 - 5.0 mEq/L.
    * If K+ > 5.2 mEq/L: Do not administer potassium; check every 2 hours.

Phase 3: Insulin Administration
  - Regular insulin infusion at 0.1 units/kg/hr.
  - Target glucose decline: 50 - 75 mg/dL per hour.
  - Continue insulin infusion until DKA resolution criteria are met:
    (Glucose < 200 mg/dL, serum bicarbonate >= 18 mEq/L, venous pH > 7.30, and anion gap normalized <= 12 mEq/L).
  - Administer basal subcutaneous insulin 2 hours prior to discontinuing intravenous insulin infusion.
```
"""
    return (
        template
        .replace("__GLUCOSE__", str(glucose))
        .replace("__KETONES__", str(ketones))
        .replace("__NA__", str(na_level))
        .replace("__K_LEVEL__", str(k_level))
        .replace("__CORRECTED_NA__", str(corrected_na))
        .replace("__SERUM_OSM__", str(serum_osm))
    )


# ==============================================================================
# Batch Generation & CLI
# ==============================================================================

GENERATORS = [
    (generate_cardiology_case, "cardiology_case"),
    (generate_pharmacokinetics_analysis, "pharmacology_pk_pd"),
    (generate_molecular_genetics_analysis, "molecular_genetics"),
    (generate_critical_care_analysis, "critical_care_shock"),
    (generate_neurology_analysis, "neurology_stroke"),
    (generate_nephrology_acid_base_analysis, "nephrology_acid_base"),
    (generate_infectious_disease_analysis, "infectious_disease"),
    (generate_endocrinology_analysis, "endocrinology_dka"),
]


def generate_batch(count: int, seed: int = 42) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """Generate a stream of clinical medicine, pharmacology, and genetics records."""
    rng = random.Random(seed)
    for i in range(count):
        gen_fn, subspecialty = rng.choice(GENERATORS)
        text = gen_fn(rng)
        meta = {
            "record_index": i + 1,
            "seed": seed + i,
        }
        yield text, subspecialty, meta


def write_partitioned_dataset(
    output_dir: Path,
    count: int = 20_000,
    max_file_mb: float = 28.0,
    seed: int = 1337,
) -> List[Path]:
    """Generate partitioned base medical, pharmacology, and genetics pretraining documents."""
    output_dir = Path(output_dir)
    max_bytes = int(max_file_mb * 1024 * 1024)
    writer = ShardedMedicineWriter(output_dir, prefix="medicine_part", max_bytes=max_bytes)

    print(f"Generating {count:,} Clinical Medicine, Pharmacology & Molecular Genetics documents into {output_dir}...")
    for idx, (text, subspecialty, meta) in enumerate(generate_batch(count, seed=seed), start=1):
        writer.write_record(text, subspecialty, metadata=meta)
        if idx % 2000 == 0 or idx == count:
            mb_written = writer.total_bytes / (1024 * 1024)
            print(f"  Progress: {idx:,} / {count:,} records processed ({mb_written:.2f} MB written)...")

    writer.close()

    manifest = {
        "version": "v1.0-medicine-base-pretraining",
        "domain": "clinical_medicine_pharmacology_genetics",
        "total_records": writer.total_records,
        "total_bytes": writer.total_bytes,
        "total_tokens_est": writer.total_bytes // 4,
        "partitions_count": len(writer.written_files),
        "partitions": [f.name for f in writer.written_files],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done! Created {len(writer.written_files)} partitions in {output_dir}")
    print(f"Manifest written to {manifest_path}")
    return writer.written_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Clinical Medicine, Pharmacology & Genetics Base Pretraining Data")
    parser.add_argument("--output-dir", type=str, default=r"E:\AI_Projects\dataset\medicine_pretraining", help="Output directory")
    parser.add_argument("--count", type=int, default=20_000, help="Number of records to generate")
    parser.add_argument("--max-partition-mb", type=float, default=28.0, help="Max MB per partition")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    write_partitioned_dataset(
        output_dir=Path(args.output_dir),
        count=args.count,
        max_file_mb=args.max_partition_mb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
