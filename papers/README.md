# Papers

A reading log of ML/AI papers. Each entry links to the arxiv page and to personal notes with analysis, key ideas, and implementation thoughts.

## Structure

```
papers/
├── README.md        ← this index
└── notes/
    └── <slug>.md    ← notes for each paper
```

Notes are named by a short slug (e.g., `attention-is-all-you-need.md`). The index below tracks what has been read, when, and links out.

---

## Index

### Foundational Architectures

| Title | Year | Description | Notes | Date Read |
|-------|------|-------------|-------|-----------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Introduces the Transformer architecture — self-attention replaces recurrence entirely, enabling parallelism and setting the blueprint for modern LLMs. | [notes](notes/attention-is-all-you-need.md) | — |

### Training & Optimization

| Title | Year | Description | Notes | Date Read |
|-------|------|-------------|-------|-----------|

### Vision & Multimodal

| Title | Year | Description | Notes | Date Read |
|-------|------|-------------|-------|-----------|
| [VLDrive: Vision-Augmented Lightweight MLLMs for Efficient Language-grounded Autonomous Driving](https://arxiv.org/abs/2511.06256) | 2025 | Lightweight MLLM for autonomous driving using cycle-consistent dynamic visual pruning and distance-decoupled instruction attention, cutting parameters 81% (7B→1.3B) while improving closed-loop driving score in CARLA. | [notes](notes/vldrive.md) | — |

### Language Models & Scaling

| Title | Year | Description | Notes | Date Read |
|-------|------|-------------|-------|-----------|

### Reinforcement Learning

| Title | Year | Description | Notes | Date Read |
|-------|------|-------------|-------|-----------|


### Efficient ML

| Title | Year | Description | Notes | Date Read |
|-------|------|-------------|-------|-----------|
|[Efficient Deep Learning: A Survey](https://arxiv.org/abs/2106.08962) | 2021 | Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better | - | - |
| [A Survey of Quantization Methods](https://arxiv.org/abs/2103.13630) | 2021 | A Survey of Quantization Methods for Efficient Neural Network Inference | - | - |
| [EfficientDet](https://arxiv.org/abs/1911.09070) | 2020 | EfficientDet: Scalable and Efficient Object Detection | - | - |

---

## How to Add a Paper

1. Add a row to the relevant section above with the arxiv link, year, and a 1–2 sentence description.
2. Create `notes/<slug>.md` using the template below.
3. Fill in the **Date Read** column once finished.

### Notes Template

```markdown
# Paper Title

**Authors:** First Author, Second Author, et al.
**Year:** YYYY
**ArXiv:** https://arxiv.org/abs/XXXX.XXXXX

## Summary

One paragraph covering the core idea and contribution.

## Key Ideas

- Bullet 1
- Bullet 2

## Method

How it works — architecture, training objective, key equations.

## Results

What they demonstrated and on which benchmarks.

## Strengths

## Weaknesses / Limitations

## My Take

Personal notes, intuitions, and connections to other work.

## Questions

Open questions or things to follow up on.
```
