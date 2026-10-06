# Student Performance Data Dictionary

Source: UCI Student Performance dataset
Source URL: https://archive.ics.uci.edu/dataset/320/student+performance
File: student-mat.csv
Subject: Mathematics
Prediction time: After the second assessment period
Target: Final grade G3

| Field | Meaning | Allowed values | Role |
|---|---|---|---|
| G1 | First-period grade | Integer 0–20 | Input |
| G2 | Second-period grade | Integer 0–20 | Input |
| studytime | Weekly study-time category | Integer 1–4 | Input |
| G3 | Final grade | Integer 0–20 | Target |

Study-time categories:
1 = Less than 2 hours
2 = 2 to 5 hours
3 = 5 to 10 hours
4 = More than 10 hours

Only G1, G2 and studytime are model inputs.
G3 must never be included among prediction inputs.

The raw CSV is separated by semicolons.

This public dataset concerns secondary-school students in Portugal.
Results are not validated predictions for our college.