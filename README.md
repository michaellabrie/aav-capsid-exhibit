# AAV Capsid Assembly Lab

**Interactive neural network exhibit · CAI5735 Applied Data Science in Health**

This is a working, self-contained static website for the course's [Build an interactive health data exhibit](https://snsie.github.io/neural-networks/exercise.html) exercise. The exhibit explores neural-network predictions on **synthetic**, AAV-themed capsid assembly scenarios, test-set trade-offs, and the difference between a model's prediction and a real biological observation.

> **Live exhibit:** Add your published GitHub Pages link here after enabling Pages.
>
> **Important:** The dataset contains invented teaching scenarios and synthetic assembly labels, not laboratory or clinical observations. Nothing in the site demonstrates real AAV serotype behavior or causal biology.

## What a visitor can do

1. **Experiment workbench:** Adjust serotype, AAP status, VP interface contact score, VP stability index, pH, ionic strength, or VP concentration and obtain a real forward-pass prediction from the trained model's exported weights. Compare the same inputs with AAP present versus absent.
2. **Threshold lab:** Move the classification threshold; see accuracy, precision, recall, F1 and a dynamically recalculated confusion matrix for 900 held-out test examples. The slider *does not* retrain the network.
3. **Mistake microscope:** Investigate a genuine held-out false positive and false negative; load the case inputs into the workbench.
4. **WebGPU lab assistant:** Click **Enable chat** to download a small pretrained language model through WebLLM and ask questions about an exhibit guide and live metrics. Chat is optional and may be unavailable on devices without WebGPU.

## How to run (no installation for the website)

Use **VS Code → File → Open Folder → `aav_capsid_exhibit`**, then open the VS Code terminal and run:

```bash
python3 -m http.server 8000
```

Open **http://localhost:8000/** in your browser. On Windows, use `py -m http.server 8000` if `python3` is unavailable. Stop the server with Ctrl+C. Do **not** double-click `index.html`: most browsers block `fetch()` from `file://`.

Alternatively, publish the folder directly with GitHub Pages; it requires **no Python, Node.js, build step, API key, or backend**.

## GitHub Pages publication — fastest way to produce the assignment link

1. Sign into [GitHub](https://github.com/), select **New repository**, and create a **public** repository named `aav-capsid-exhibit`. You do not need a GitHub template or other initialization options.
2. In the terminal with the `aav_capsid_exhibit` directory open, run:

   ```bash
   git init
   git add .
   git commit -m "Complete interactive AAV neural network exhibit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/aav-capsid-exhibit.git
   git push -u origin main
   ```

   **Replace `YOUR-USERNAME` with your GitHub username.** GitHub may prompt you to sign in. If the repo already has Git set up or a remote configured, skip the matching `git init`/`git remote add` step.
3. On the GitHub repository, open **Settings → Pages**, choose **Deploy from a branch**, set the branch to **main** and the folder to **/(root)**, and press **Save**.
4. After the deployment finishes, your live website will normally be at:

   `https://YOUR-USERNAME.github.io/aav-capsid-exhibit/`

   This URL is only a *pattern* until GitHub Pages publishes the repository. Open it in a **private/incognito browser window** to verify the live exhibit works.
5. **Submit the working public website URL in Canvas.** The course exercise states the deliverable is a website link that opens directly to the exhibit. If your instructor also requests the repo link, include `https://github.com/YOUR-USERNAME/aav-capsid-exhibit`.

**GitHub alternative:** Without Git commands, create the empty GitHub repo, click **Add file → Upload files**, and drag the *contents* of `aav_capsid_exhibit` into it. Confirm the `index.html` file appears at the repository root and folders `assets`, `data`, `model` and `scripts` were preserved. Then enable Pages as above.

## Model training and reproducible data

- Dataset: **6,000 synthetic scenarios** (seed `20261119`). Train `4,200`, validation `900`, untouched test `900`. The exact generating equation appears in [`scripts/train_model.py`](scripts/train_model.py).
- Inputs: AAP presence, interface index, VP stability index, pH, ionic strength, VP concentration, and one-hot AAV2/AAV5/AAV8/AAV9 category (`10` total numerical columns).
- Neural network: `scikit-learn` `MLPClassifier`, **10 → 14 → 8 → 1**, ReLU hidden layers, sigmoid output, Adam optimizer, train-only standardization.
- Validation chooses the classification threshold by F1 score; held-out test results are computed at the chosen threshold.
- Baseline: predict the majority class of the **training** examples for every test case (this baseline is *not* a neural network).
- Data for the interactive case files and threshold explorer are exclusively from the held-out test split.

### Frozen test results

| Metric | Result |
|---|---:|
| Neural-network test accuracy | **76.67%** |
| Majority-class baseline test accuracy | **52.33%** |
| Test precision | **70.54%** |
| Test recall | **87.65%** |
| Test F1 | **78.17%** |
| Test ROC AUC | **86.11%** |
| Validation-selected threshold | **0.315** |
| True positives / false positives | **376 / 157** |
| False negatives / true negatives | **53 / 314** |

These values are from an actual trained network and generated test examples; they are **not real experimental accuracy estimates**. Metrics are also stored in [`model/metrics.json`](model/metrics.json).

### Reproduce training (optional)

Python 3 with numpy and scikit-learn:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/train_model.py
```

This rewrites the generated training CSV, held-out examples, exported model weights and metrics with a fixed seed and split. It is **not needed** to view or publish the website.

## Chatbot and privacy

The chat uses the official [WebLLM](https://webllm.mlc.ai/docs/user/get_started.html) client, dynamically imported **only when the visitor clicks Enable chat**. It chooses the WebLLM-bundled SmolLM2-360M instruct model when available, or a small compatible alternative. Downloading the first model may transfer a few hundred MB; GPU/driver availability varies. Chat inference runs locally, without sending questions to an external LLM inference API. Models, libraries and weights are downloaded from third-party hosts. If the browser lacks WebGPU or the download fails, the rest of the website remains functional. If a guide answer is absent, the model is instructed to say the guide lacks the information. **Small LMs can still hallucinate**: verify answers against the data and metrics.

### Three chatbot test questions

- What does the model predict, and which inputs does it use?
- How does the network compare with the baseline on the test set?
- Does the exhibit prove that AAP causes capsid assembly in real experiments?

The expected answers, respectively, are a synthetic assembled/not-assembled probability based on 10 features; **76.67% versus 52.33% test accuracy**; and **no**, because the labels and AAP effects are simulated.

## Limitations and error analysis

The training generator intentionally encodes simplified serotype- and AAP-associated effects, environmental effects and Bernoulli sampling noise. A real biological model would require independent experimental measurements, carefully designed controls, and external validation. Neither synthetic correlation nor an AAP toggling contrast establishes physical causality. Actual capsid assembly, vector packaging, infectivity and clinical effectiveness are different measurements. The model has false positives and false negatives on its own synthetic test distribution, and thresholds expose the consequences of choosing one kind of mistake over another. For example, the **false-negative** example in the Mistake Microscope represents a held-out case labeled assembled even though the model output was below the frozen threshold; repeated simulated observations for similar inputs could reduce uncertainty about the underlying probability. More informative experimentally measured features might help on real problems, but no such benefit was tested here.

## Files

```text
aav_capsid_exhibit/
├── index.html                      # Static website (GitHub Pages entry point)
├── assets/
│   ├── app.js                      # Browser inference, interaction, chatbot
│   ├── styles.css                  # Responsive laboratory-styled interface
│   └── favicon.svg                 # Original geometric icon
├── model/
│   ├── network.json                # Trained, exported network and scaler
│   └── metrics.json                # Verified train/validation/test metrics
├── data/
│   ├── synthetic_aav_assembly.csv  # All 6,000 labeled synthetic scenarios
│   └── heldout_test.json           # Only the 900 test cases (for exhibit)
├── scripts/
│   ├── train_model.py              # Reproducible generator and training
│   └── verify_exhibit.py           # Dependency-free exported-model QA              # Full reproducible data + model pipeline
├── guide.txt                       # Context given to the local chatbot
├── requirements.txt
├── .nojekyll
└── README.md
```

## Pair roles (course exercise)

- **Model / evaluation:** Generated simulated data, trained the network, compared its held-out test results with the baseline, exported model weights and reviewed a false negative.
- **Website / chatbot:** Built the responsive controls and test-set explorer, integrated WebLLM, wrote the exhibit guide and deployment workflow.

These describe workstreams rather than attributing contributions to a particular person. Add your partner's name and actual contributions here if your instructor requires individual attribution.

## Credits

This is a synthetic, student educational demo for the class activity at [snsie.github.io](https://snsie.github.io/neural-networks/exercise.html), with an AAV theme. For biological background on AAP dependence, see [AAP and capsid assembly](https://pmc.ncbi.nlm.nih.gov/articles/PMC5983388/). This link is background reading, **not** the source of simulation labels.

The code and original synthetic dataset may be reused under the included MIT license. Third-party WebLLM models and packages have their own licenses.
