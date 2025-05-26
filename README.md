# 🧪 Evaluating the Impact of Code Review on Bug Reduction in Open Source Projects

This project investigates whether open-source projects that enforce mandatory code reviews exhibit fewer bugs, faster resolution times, and better software quality than those that do not. Using real GitHub data and statistical analysis, this study compares multiple repositories across key metrics such as **bug density**, **review depth**, and **bug resolution time**.

---

## 📊 Project Summary

- **Objective:** Understand the effect of code reviews on software quality in OSS.
- **Methodology:** Extract data using GitHub API, calculate quality metrics, and analyze using statistical tests.
- **Key Metrics:**
  - 🐛 Bug Density
  - 💬 Review Depth (comments per PR)
  - ⏱️ Bug Resolution Time

---

## 🔍 Repositories Analyzed

| Repository              | Code Review Enforced |
|-------------------------|----------------------|
| `facebook/react`        | ✅ Yes               |
| `microsoft/vscode`      | ✅ Yes               |
| `tensorflow/tensorflow` | ✅ Yes               |
| `torvalds/linux`        | ❌ No                |
| `git/git`               | ❌ No                |
| `vim/vim`               | ❌ No                |

---

## 📁 Files in This Repository

| File                             | Description                                                  |
|----------------------------------|--------------------------------------------------------------|
| `enhanced_code_review_analysis.py` | Python script to collect, process, and analyze GitHub data   |
| `enhanced_code_review_analysis.csv` | Processed dataset containing bug density, review depth, etc. |
| `bug_resolution_times.png`      | Bar chart of average bug resolution time by repository       |
| `enhanced_code_review_analysis.png` | Composite visualization of all key metrics                  |
| `README.md`                     | This documentation file                                      |

---

## 📈 Key Findings

- ✅ **Lower Bug Density** in projects with mandatory code reviews.
- 💬 **Higher Review Depth** where PRs and structured reviews are enforced.
- ⏱️ **Faster Bug Resolution** observed in some reviewed projects (e.g., React, VSCode).

> 📌 _Statistical tests (T-test & Mann-Whitney U) confirm significant differences in bug density and review depth._

---

## 📚 Technologies Used

- 🐍 Python 3
- 🧰 [PyGithub](https://pygithub.readthedocs.io/)
- 📊 Matplotlib & Seaborn
- 📁 Pandas & NumPy
- 📎 GitHub REST API

---

## 📌 How to Run

1. **Install dependencies:**

    pip install -r requirements.txt

2. **Set Up your .env file:**

    GITHUB_TOKEN=your_personal_access_token

3. **Run the operation:**

    python main.py

---
