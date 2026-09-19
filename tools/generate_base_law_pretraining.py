"""Frontier Law, Governance, Contracts & Regulation Base Pretraining Generator.

Generates high-fidelity, authentic legal agreements, statutory frameworks,
regulatory compliance specifications, and legal analyses formatted for base pre-training.

Disciplines:
1. Commercial Contracts & MSAs (Indemnification, Limitation of Liability, IP assignment, Force Majeure)
2. Global Regulatory Compliance (GDPR, EU AI Act, SEC Regulations, HIPAA, SOX)
3. Jurisprudence & IRAC Legal Analyses (Breach of contract, Tort liability, Fiduciary duties)
4. Intellectual Property & Licensing (Patent claims, Trade secrets, Apache/GPL license architectures)

Enforces strict partition sizing: files are partitioned to stay <= 28.0 MB (29,360,128 bytes).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_PARTITION_BYTES = 28 * 1024 * 1024  # 28 MB ceiling


# ==============================================================================
# Partitioned Writer
# ==============================================================================

class ShardedLawWriter:
    """Writes JSONL base pretraining records into partitioned files under 28 MB."""

    def __init__(self, output_dir: Path, prefix: str = "base_law_part", max_bytes: int = MAX_PARTITION_BYTES) -> None:
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

    def write_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        doc = {
            "text": text,
            "meta": meta or {}
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


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ==============================================================================
# Domain 1: Commercial Contracts & MSAs
# ==============================================================================

def gen_contract_msa_indemnification(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    jurisdiction = rng.choice(["State of Delaware", "State of New York", "England and Wales"])
    cap_multiple = rng.choice(["two (2x)", "three (3x)", "twelve (12) months of"])
    template = r"""# MASTER SERVICES AGREEMENT (MSA)
## Standard Corporate Enterprise Framework | Governing Law: __JURISDICTION__

This Master Services Agreement ("Agreement") is made effective as of the Effective Date by and between Provider and Customer (each a "Party", and collectively the "Parties").

### SECTION 8: INDEMNIFICATION

**8.1 Provider Intellectual Property Indemnity.**
Provider shall defend, indemnify, and hold harmless Customer, its corporate affiliates, and their respective directors, officers, employees, and agents (collectively, the "Customer Indemnitees") from and against any and all third-party claims, suits, actions, or proceedings (each, a "Claim") alleging that the Services or Deliverables, when used in accordance with this Agreement and the applicable Statement of Work ("SOW"), infringe, misappropriate, or violate any patent, copyright, trademark, trade secret, or other proprietary intellectual property right of any third party. Provider shall pay all damages, liabilities, costs, and expenses (including reasonable outside attorneys' fees) finally awarded by a court of competent jurisdiction or agreed upon in a formal written settlement approved in advance by Provider.

**8.2 Exclusions.**
Provider's obligations under Section 8.1 shall not apply to the extent any Claim arises from:
(a) Modifications to the Services or Deliverables made by any party other than Provider without Provider's express written authorization;
(b) Combination, operation, or use of the Services with equipment, devices, software, hardware, or data not supplied or specified by Provider;
(c) Customer's continuing allegedly infringing activity after being provided with modifications, replacements, or updates that would have avoided the alleged infringement without material loss of functionality; or
(d) Compliance by Provider with designs, specifications, or directions furnished directly by Customer.

**8.3 Customer Indemnification.**
Customer shall defend, indemnify, and hold harmless Provider, its affiliates, and their respective officers, directors, and employees from and against any third-party Claims arising out of or relating to:
(a) Customer Data or third-party content provided by Customer for ingestion or processing by the Services; or
(b) Customer's material breach of Section 3 (Acceptable Use Policy) or applicable data protection regulations.

**8.4 Indemnification Procedures.**
Each Party's indemnification obligations under this Section 8 are expressly conditioned upon the indemnified Party:
(i) Providing prompt written notice of the Claim to the indemnifying Party (provided that failure to provide prompt notice shall relieve the indemnifying Party of its obligations only to the extent it is materially prejudiced thereby);
(ii) Granting the indemnifying Party sole control of the defense and settlement of the Claim (provided that the indemnifying Party shall not settle any Claim that admits liability or imposes monetary or injunctive obligations on the indemnified Party without its prior written consent); and
(iii) Providing reasonable cooperation and assistance to the indemnifying Party, at the indemnifying Party's expense.

### SECTION 9: LIMITATION OF LIABILITY

**9.1 Consequential Damages Waiver.**
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL EITHER PARTY, ITS AFFILIATES, OR THEIR RESPECTIVE LICENSORS BE LIABLE UNDER ANY LEGAL THEORY (WHETHER IN CONTRACT, TORT, NEGLIGENCE, STRICT LIABILITY, OR OTHERWISE) FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, PUNITIVE, EXEMPLARY, OR MULTIPLIED DAMAGES, INCLUDING LOSS OF PROFITS, LOSS OF REVENUE, LOSS OF GOODWILL, LOSS OF USE, BUSINESS INTERRUPTION, OR PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, REGARDLESS OF WHETHER SUCH PARTY WAS ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

**9.2 Aggregate Liability Cap.**
EXCEPT FOR THE EXCLUDED CLAIMS IDENTIFIED IN SECTION 9.3, EACH PARTY'S TOTAL AGGREGATE LIABILITY ARISING OUT OF OR RELATING TO THIS AGREEMENT, THE SERVICES, OR ANY SOW SHALL BE STRICTLY LIMITED TO __CAP_MULTIPLE__ THE FEES ACTUALLY PAID OR PAYABLE BY CUSTOMER UNDER THE APPLICABLE SOW DURING THE TWELVE (12) MONTH PERIOD IMMEDIATELY PRECEDING THE FIRST INCIDENT GIVING RISE TO LIABILITY.

**9.3 Excluded Claims.**
The waivers and limitations set forth in Sections 9.1 and 9.2 shall not apply to:
(a) Either Party's breach of Section 7 (Confidentiality), excluding claims relating to data breaches addressed under Section 9.4;
(b) Either Party's indemnification obligations under Section 8;
(c) Damages arising from a Party's gross negligence, willful misconduct, or intentional fraud; or
(d) Customer's obligation to pay undisputed Fees properly accrued under any active SOW.
"""
    text = template.replace("__JURISDICTION__", jurisdiction).replace("__CAP_MULTIPLE__", cap_multiple)
    meta = {
        "domain": "law_governance",
        "subdomain": "commercial_contracts",
        "topic": "msa_indemnification_liability",
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 2: Global Regulatory Compliance
# ==============================================================================

def gen_gdpr_data_processing_agreement(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    dpo_city = rng.choice(["Dublin, Ireland", "Frankfurt, Germany", "Amsterdam, Netherlands"])
    template = r"""# REGULATORY COMPLIANCE FRAMEWORK: GENERAL DATA PROTECTION REGULATION (GDPR)
## Standard Contractual Clauses & Data Processing Addendum (DPA) under Regulation (EU) 2016/679

### ARTICLE 1: RECITALS & SCOPE
This Data Processing Addendum ("DPA") supplements the Principal Agreement between Data Controller and Data Processor. It regulates the processing of Personal Data in compliance with Regulation (EU) 2016/679 of the European Parliament and of the Council (General Data Protection Regulation, "GDPR").
The supervisory authority of competent jurisdiction shall be determined in accordance with the Lead Supervisory Authority in __DPO_CITY__.

### ARTICLE 2: INSTRUCTIONS & NATURE OF PROCESSING
1. **Scope and Purpose:** Processor shall process Personal Data solely on documented instructions from Controller, including with regard to transfers of Personal Data to a third country or international organization, unless required to do so by European Union or Member State law to which Processor is subject.
2. **Duty of Notification:** Processor shall immediately inform Controller if, in its reasonable opinion, an instruction infringes the GDPR or other Union or Member State data protection provisions (Article 28(3)(h)).

### ARTICLE 3: TECHNICAL AND ORGANIZATIONAL MEASURES (TOMs) (Article 32)
Processor shall implement appropriate technical and organizational measures to ensure a level of security appropriate to the risk, taking into account the state of the art, the costs of implementation, and the nature, scope, context, and purposes of processing, including as appropriate:
(a) The pseudonymisation and AES-256 encryption of Personal Data at rest and TLS 1.3 encryption in transit;
(b) The ability to ensure the ongoing confidentiality, integrity, availability, and resilience of processing systems and services;
(c) The ability to restore the availability and access to Personal Data in a timely manner in the event of a physical or technical incident; and
(d) A process for regularly testing, assessing, and evaluating the effectiveness of technical and organizational measures for ensuring the security of the processing.

### ARTICLE 4: SUB-PROCESSORS (Article 28(2) & 28(4))
1. **Prior Authorization:** Processor shall not engage another processor ("Sub-processor") without prior specific or general written authorization of Controller.
2. **Objection Period:** In the case of general written authorization, Processor shall notify Controller at least thirty (30) days in advance of any intended changes concerning the addition or replacement of Sub-processors, thereby giving Controller the opportunity to object to such changes on reasonable data protection grounds.
3. **Flow-Down Obligations:** Where Processor engages a Sub-processor, it shall do so by way of a written contract imposing the same data protection obligations as set out in this DPA. Processor remains fully liable to Controller for the performance of the Sub-processor's obligations.

### ARTICLE 5: DATA SUBJECT RIGHTS & ASSISTANCE (Articles 15–22)
Taking into account the nature of the processing, Processor shall assist Controller by appropriate technical and organizational measures, insofar as this is possible, for the fulfilment of Controller's obligation to respond to requests for exercising the data subject's rights laid down in Chapter III GDPR, specifically:
- Article 15: Right of access by the data subject;
- Article 16: Right to rectification;
- Article 17: Right to erasure ('right to be forgotten');
- Article 18: Right to restriction of processing;
- Article 20: Right to data portability; and
- Article 21: Right to object.

### ARTICLE 6: PERSONAL DATA BREACH NOTIFICATION (Article 33)
1. **Notice Window:** Processor shall notify Controller without undue delay and, in any event, within forty-eight (48) hours after becoming aware of a Personal Data Breach.
2. **Breach Particulars:** The notification shall contain at least:
   (a) A description of the nature of the Personal Data Breach including where possible, the categories and approximate number of data subjects concerned and the categories and approximate number of personal data records concerned;
   (b) The name and contact details of the Data Protection Officer (DPO) or other contact point where more information can be obtained;
   (c) A description of the likely consequences of the Personal Data Breach; and
   (d) A description of the measures taken or proposed to be taken by Processor to address the Personal Data Breach, including, where appropriate, measures to mitigate its possible adverse effects.
"""
    text = template.replace("__DPO_CITY__", dpo_city)
    meta = {
        "domain": "law_governance",
        "subdomain": "regulatory_compliance",
        "topic": "gdpr_dpa_compliance",
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


def gen_eu_ai_act_compliance_spec(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    risk_tier = rng.choice(["High-Risk AI System (Annex III)", "General Purpose AI Model with Systemic Risk (GPAI)"])
    template = r"""# REGULATORY COMPLIANCE SPECIFICATION: EUROPEAN UNION ARTIFICIAL INTELLIGENCE ACT
## Conformity Assessment & Risk Governance under Regulation (EU) 2024/1689
**Classification:** __RISK_TIER__

### SECTION 1: CLASSIFICATION RULES & LEGAL SCOPE
Under the EU AI Act (Regulation (EU) 2024/1689), artificial intelligence systems are regulated through a risk-based hierarchy:
1. **Prohibited AI Practices (Article 5):** Cognitive behavioral manipulation, untargeted scraping of facial images from the internet or CCTV, social scoring, biometric categorization deducing sensitive attributes, and real-time remote biometric identification in publicly accessible spaces for law enforcement (subject to narrow judicial exceptions).
2. **High-Risk AI Systems (Articles 6–51):** Systems used as safety components of regulated products (Annex I) or systems listed in Annex III (biometrics, critical infrastructure, education, employment, access to essential services, law enforcement, migration, administration of justice).
3. **General Purpose AI (GPAI) Models (Articles 51–56):** Models exhibiting high generality capable of performing a wide range of tasks, categorized into standard GPAI and GPAI with systemic risk (cumulative training compute > 10^25 FLOPs).

### SECTION 2: MANDATORY REQUIREMENTS FOR HIGH-RISK SYSTEMS (Chapter III, Section 2)

#### Article 9: Risk Management System
A risk management system shall be established, implemented, documented, and maintained as a continuous iterative process throughout the entire lifecycle of a high-risk AI system. It shall require:
(a) Identification and analysis of the known and reasonably foreseeable risks that the system may pose to health, safety, or fundamental rights;
(b) Evaluation of risks that may emerge when the high-risk AI system is used in accordance with its intended purpose and under conditions of reasonably foreseeable misuse;
(c) Adoption of targeted risk management measures pursuant to the principle of safety-by-design: elimination or reduction of risks through design, implementation of mitigation and control measures, and provision of adequate technical instructions.

#### Article 10: Data Governance and Data Management
High-risk AI systems which make use of techniques involving the training of models with data shall be developed on the basis of training, validation, and testing datasets that meet strict quality criteria:
- Data governance practices must examine design choices, data collection, data preparation (cleaning, enrichment, aggregation), and assessment of bias;
- Datasets must be relevant, sufficiently representative, and, to the best extent possible, free of errors and complete in view of the intended purpose;
- Datasets must take into account the specific geographical, contextual, behavioral, or functional setting within which the AI system is intended to be used.

#### Article 11 & Annex IV: Technical Documentation
Technical documentation shall be drawn up before the system is placed on the market or put into service and kept up-to-date. Documentation must contain:
1. General description of the AI system, intended purpose, versions, and hardware interaction;
2. Detailed description of system architecture, algorithmic logic, training methodologies, hyperparameters, validation benchmarks, and test protocols;
3. Monitoring, functioning, and control specifications, including metrics for accuracy, robustness, and cybersecurity.

#### Article 14: Human Oversight
High-risk AI systems shall be designed and developed in such a way, including with appropriate human-machine interface tools, that they can be effectively overseen by natural persons during the period in which they are in use:
- Overseers must understand the capacities and limitations of the high-risk AI system and be able to monitor its operation to detect signs of anomalies, dysfunction, and unexpected performance;
- System design must enable the human overseer to decide, in any particular situation, not to use the system or to disregard, override, or reverse the system's output;
- Must include a "stop" button or equivalent kill-switch mechanism to safely halt the system without physical harm.

### SECTION 3: PENALTIES & ENFORCEMENT (Article 99)
Member States shall lay down rules on penalties:
- Non-compliance with Article 5 (Prohibited practices): Fines up to **€35,000,000 or 7% of total worldwide annual turnover** for the preceding financial year, whichever is higher;
- Non-compliance with Chapter III (High-risk requirements): Fines up to **€15,000,000 or 3% of total worldwide annual turnover**;
- Supplying incorrect, incomplete, or misleading information to notified bodies: Fines up to **€7,500,000 or 1.5% of total worldwide annual turnover**.
"""
    text = template.replace("__RISK_TIER__", risk_tier)
    meta = {
        "domain": "law_governance",
        "subdomain": "regulatory_compliance",
        "topic": "eu_ai_act_compliance",
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 3: Jurisprudence & IRAC Legal Analyses
# ==============================================================================

def gen_legal_irac_corporate_governance(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    duty = rng.choice(["Duty of Loyalty and the Business Judgment Rule", "Duty of Care and Oversight Liability under Caremark"])
    template = r"""# CORPORATE JURISPRUDENCE: IRAC LEGAL MEMORANDUM
## Subject: Fiduciary Duties of Corporate Directors under Delaware General Corporation Law (DGCL)
### Focus Area: __DUTY__

### 1. ISSUE
Whether the board of directors of a publicly traded Delaware corporation breached their fiduciary duties to the corporation and its stockholders by approving a contested interested-party merger without conditioning the transaction upon the dual procedural protections established in *Kahn v. M&F Worldwide Corp. (MFW)*, and whether the board's decision is shielded by the deferential Business Judgment Rule or subject to the rigorous Entire Fairness standard of judicial review.

### 2. RULE OF LAW

**The Business Judgment Rule (BJR):**
Under Delaware corporate law, the business judgment rule is a powerful presumption that in making a business decision, the directors of a corporation acted on an informed basis, in good faith, and in the honest belief that the action taken was in the best interests of the company (*Aronson v. Lewis*, 473 A.2d 805 (Del. 1984)). Where the BJR applies, a court will not substitute its judgment for that of the board if the decision can be attributed to any rational business purpose.

**Fiduciary Duties of Care and Loyalty:**
1. **Duty of Care:** Requires directors to inform themselves, prior to making a business decision, of all material information reasonably available to them (*Smith v. Van Gorkom*, 488 A.2d 858 (Del. 1985)). Gross negligence is the standard for a breach of the duty of care.
2. **Duty of Loyalty:** Mandates that the best interests of the corporation and its shareholders take precedence over any personal interest or benefit not shared generally by the stockholders (*Guth v. Loft, Inc.*, 5 A.2d 503 (Del. 1939)). Conflicts of interest, bad faith, and intentional dereliction of duty violate the duty of loyalty.

**Entire Fairness Review vs. MFW Cleansing:**
When a controlling stockholder stands on both sides of a transaction or extracts disparate consideration, the transaction is presumptively reviewed under Delaware's most onerous standard: **Entire Fairness**, placing the burden on the defendant fiduciaries to prove both:
(a) **Fair Dealing:** Analyzing the timing, initiation, negotiation, structuring, and disclosure to the board and stockholders; and
(b) **Fair Price:** Analyzing the economic and financial substance of the transaction (*Weinberger v. UOP, Inc.*, 457 A.2d 701 (Del. 1983)).

Under *Kahn v. M&F Worldwide Corp. (MFW)*, 88 A.3d 635 (Del. 2014), the standard of review shifts from Entire Fairness back to the deferential Business Judgment Rule *ab initio* if and only if the controller conditions the merger upfront upon both:
1. The negotiation and approval of a fully empowered, independent Special Committee of directors; and
2. The uncoerced, fully informed vote of a majority of the minority stockholders.

### 3. APPLICATION & ANALYSIS
In the instant case, while the board appointed a Special Committee to negotiate the transaction with the controlling shareholder, the controller did not irrevocably condition the transaction upon the dual protections *prior to the commencement of substantive negotiations*. Under *MFW* and *Tornetta v. Musk*, 310 A.3d 430 (Del. Ch. 2024), procedural cleansing cannot be retroactively implemented; the controller must self-disable at the threshold of the negotiation process.

Because the prerequisite condition was not met at the inception:
1. The transaction remains subject to Entire Fairness review;
2. The burden of proving fair dealing and fair price rests with the defendant directors and the controller;
3. Exculpation under DGCL § 102(b)(7) is unavailable for breaches of the duty of loyalty, meaning that monetary liability against director defendants cannot be dismissed at the pleadings stage.

### 4. CONCLUSION
The board's failure to structure the transaction under strict adherence to the *MFW* framework deprives the directors of Business Judgment Rule protection. The court must review the merger under the Entire Fairness standard. Unless the defendant directors can prove that the process reflected arm's-length negotiation and the valuation constituted fair value, the directors face individual and joint liability for breach of their fiduciary duty of loyalty.
"""
    text = template.replace("__DUTY__", duty)
    meta = {
        "domain": "law_governance",
        "subdomain": "jurisprudence_irac",
        "topic": "corporate_fiduciary_duties",
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 4: Intellectual Property & Open Source Licensing
# ==============================================================================

def gen_ip_licensing_patent_architecture(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    license_type = rng.choice(["Apache License Version 2.0 Patent Grant", "GPLv3 Incompatible Patent Retaliation Clause"])
    template = r"""# INTELLECTUAL PROPERTY & SOFTWARE LICENSING ARCHITECTURE
## Formal Analysis of Patent Retaliation and Reciprocal Licensing Mechanics
### Subject: __LICENSE_TYPE__

### 1. FOUNDATIONS OF SOFTWARE PATENT GRANTS
Modern open source software licenses address two distinct intellectual property rights:
1. **Copyright License:** Grants the right to reproduce, prepare derivative works of, publicly display, publicly perform, and distribute the software source code and binaries.
2. **Patent License:** Grants express, royalty-free, perpetual patent licenses for claims necessarily infringed by the software alone or by combination with the contribution.

### 2. SECTION 3 OF THE APACHE 2.0 LICENSE (GRANT OF PATENT LICENSE)
The patent clause of the Apache 2.0 License provides:
> "Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted."

**Defensive Termination (Patent Retaliation Clause):**
> "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed."

### 3. LEGAL MECHANICS & INVARIANTS

**1. Contributor Patent Exhaustion:**
A contributor who submits code to an Apache 2.0 project implicitly and explicitly surrenders the ability to assert their patent portfolio against downstream users of that contribution. This creates a legal patent common pool for project participants.

**2. Asymmetric vs. Symmetric Retaliation:**
- Under Apache 2.0, retaliation is **specific to the Work**. If licensee A sues licensee B alleging that the *Apache-licensed Work* infringes A's patent, A loses patent rights to that specific Work. If A sues B over an unrelated product, A does not lose its license to the Apache Work.
- In contrast, under broader reciprocal patent licenses (e.g., Common Public License / Eclipse Public License), retaliation may be triggered by *any* patent suit filed against a contributor, creating a broader defensive perimeter.

**3. Interaction with Section 11 of GNU General Public License version 3 (GPLv3):**
Under GPLv3 Section 11, each contributor grants a non-exclusive, worldwide, royalty-free patent license under the contributor's essential patent claims. Furthermore:
- **Discriminatory Patent Agreements:** If an entity enters into a patent agreement with a third party (such as a patent covenant not to sue) that shields only a subset of downstream recipients, GPLv3 Article 11 automatically extends that patent protection to all downstream recipients worldwide, neutralizing private patent carve-outs.

### 4. PRACTICAL RISK MITIGATION IN ENTERPRISE COMMERCE
1. **Inbound Contributor License Agreements (CLAs):** Enterprise software foundations mandate Signed CLAs (e.g., Apache CLA) to establish an unbroken chain of title and explicit warranties of authority to grant patent licenses.
2. **Patent Clean-Room Policies:** Engineers contributing to open-source systems must not consult external patent filings (avoiding allegations of willful infringement under 35 U.S.C. § 284, which permits treble damages).
"""
    text = template.replace("__LICENSE_TYPE__", license_type)
    meta = {
        "domain": "law_governance",
        "subdomain": "intellectual_property",
        "topic": "patent_licensing_retaliation",
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Master Generator Registry
# ==============================================================================

LAW_GENERATOR_REGISTRY: List[Callable[[random.Random], Tuple[str, Dict[str, Any]]]] = [
    gen_contract_msa_indemnification,
    gen_gdpr_data_processing_agreement,
    gen_eu_ai_act_compliance_spec,
    gen_legal_irac_corporate_governance,
    gen_ip_licensing_patent_architecture,
]


def generate_base_law_corpus(
    output_dir: Path,
    count: int = 15000,
    seed: int = 42,
    max_mb: float = 28.0,
    prefix: str = "base_law_part"
) -> Dict[str, Any]:
    """Generate partitioned base legal, regulatory, and contract pretraining documents strictly under 28 MB."""
    rng = random.Random(seed)
    max_bytes = int(max_mb * 1024 * 1024)
    writer = ShardedLawWriter(output_dir=output_dir, prefix=prefix, max_bytes=max_bytes)

    subdomain_counts: Dict[str, int] = {}
    total_tokens_est = 0

    print(f"Generating {count:,} Law, Governance, Contracts & Regulation pretraining documents into {output_dir}...")

    try:
        for i in range(count):
            gen_fn = rng.choice(LAW_GENERATOR_REGISTRY)
            text, meta = gen_fn(rng)

            writer.write_document(text=text, meta=meta)

            sub = meta.get("subdomain", "law")
            toks = meta.get("tokens", estimate_tokens(text))

            subdomain_counts[sub] = subdomain_counts.get(sub, 0) + 1
            total_tokens_est += toks

            if (i + 1) % 1000 == 0 or (i + 1) == count:
                print(f"  Progress: {i + 1:,} / {count:,} records processed...")
    finally:
        writer.close()

    manifest = {
        "version": "base-law-v1.0",
        "total_records": writer.total_records,
        "total_bytes": writer.total_bytes,
        "total_mb": writer.total_bytes / (1024 * 1024),
        "estimated_tokens": total_tokens_est,
        "partitions_count": len(writer.written_files),
        "partition_files": [f.name for f in writer.written_files],
        "subdomain_distribution": subdomain_counts,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\nCompleted! Generated {writer.total_records:,} records across {len(writer.written_files)} partitions.")
    print(f"Total Size: {manifest['total_mb']:.2f} MB (~{total_tokens_est / 1e6:.1f} M tokens)")
    print(f"Manifest written to: {manifest_path}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Law & Governance Base Pretraining Data")
    parser.add_argument("--output-dir", type=str, default=r"E:\AI_Projects\dataset\law_pretraining", help="Output directory")
    parser.add_argument("--count", type=int, default=15000, help="Number of documents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-mb", type=float, default=28.0, help="Partition ceiling in MB")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    generate_base_law_corpus(
        output_dir=out_path,
        count=args.count,
        seed=args.seed,
        max_mb=args.max_mb,
    )


if __name__ == "__main__":
    main()
