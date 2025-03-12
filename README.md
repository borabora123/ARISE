# ARISE-2025: Automated Rheumatoid Inflammatory Scoring and Evaluation

## Task Description

Rheumatoid arthritis (RA) is a chronic autoimmune disease characterized by inflammation, joint destruction, and extra-articular manifestations. Radiography is the standard imaging modality for diagnosing and monitoring joint damage in RA. However, traditional methods for evaluating radiographic progression, such as the Sharp method and its variants, are time-consuming and subjective. This hackathon focuses on developing automated solutions for joint assessment in RA using computer vision techniques.

Participants will build models to automatically score hand joints affected by RA. The task involves two key components:
1. **Joint Localization**: Accurately localize hand joints in radiographic images.
2. **Pathology Assessment**: Evaluate the severity of joint damage, specifically focusing on **erosion** and **joint space narrowing (JSN)** and predict damage scores (0-4 for JSN and 0-5 for erosion).

The goal is to develop a robust and efficient pipeline that can assist clinicians in diagnosing and monitoring RA progression, reducing subjectivity and manual effort.

---

## Evaluation Metrics

The performance of the models will be evaluated using the following metrics:

### 1. **Intersection over Union (IoU)**
IoU measures the overlap between the predicted bounding box and the ground truth bounding box. It is defined as:

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

- **Area of Overlap**: The region where the predicted and ground truth bounding boxes intersect.
- **Area of Union**: The total area covered by both the predicted and ground truth bounding boxes.

A higher IoU indicates better localization accuracy.

---

### 2. **Balanced Accuracy**
Balanced Accuracy is a metric used to evaluate the performance of a classification model, especially in cases where the classes are imbalanced. It is the average of recall (sensitivity) obtained on each class, ensuring that the performance metric is not biased toward the majority class.

Balanced Accuracy is defined as:

$$
\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}} + \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}} \right)
$$

---

### 3. **Final Metric: IoU × Accuracy**
The final evaluation metric is the product of **IoU** and **Accuracy**. This combined metric ensures that models achieve both precise localization and accurate pathology assessment. It is defined as:

$$
\text{Final Metric} = \text{IoU} \times \text{Accuracy}
$$

The final score ranges between 0 and 1, where higher values indicate better overall performance.

---
