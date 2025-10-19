# 🧩 MedXplain — Neuro-Symbolic Explainability for Vision–Language Models in Medicine

### 📘 Course: Computer Vision — Midterm Project (CAU, Fall 2025)
Instructors: Dr. I. Atadjanov & Dr. B. Kiani  
**Team:** MedXplain  
- Inoyatov Ulughbek  
- Shahzod Numanjonov  
- Fayzulloh Asatullayev  

---

## 🧠 Project Overview
Modern vision–language models (VLMs) such as **BLIP-2** and **LLaVA-Med** achieve strong visual–text alignment in medical imaging,  
but their internal reasoning remains opaque to clinicians.  
This project aims to design a **neuro-symbolic explainability layer** that transforms model attention patterns into  
structured, interpretable **concept graphs** linking lesions, organs, and diagnostic phrases.

We focus on **radiology images (chest X-rays)** from open datasets and evaluate how symbolic reasoning improves the **faithfulness and interpretability** of model predictions without reducing diagnostic accuracy.

---

## 🎯 Objectives
1. Build a lightweight explainability framework for BLIP-2 / LLaVA-Med.  
2. Convert image–text attention maps into **symbolic causal graphs**.  
3. Quantitatively evaluate model **faithfulness and localization** (pointing-game, deletion/insertion metrics).  
4. Provide clinicians with **visual and textual explanations** they can interpret.

---

## 🧩 Methodology

**Pipeline Overview:**
