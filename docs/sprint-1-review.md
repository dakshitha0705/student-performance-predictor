# Sprint 1 review and retrospective

## Review record
- Sprint: SPP Sprint 1 - Data Model API (started 7 Oct 2026, closing 8 Oct 2026)
- Goal: deliver validated and versioned data, evaluated baselines, a reproducible model package and a working prediction API.
- Review held: live, over the team call, on the evenings of 7 Oct 2026 and 8 Oct 2026. Exact clock times were not recorded.
- Attendees: Dakshitha (Scrum master), Divya, Prabhanjan, Chaitanya.

## Work demonstrated
| Issue | Item | Demonstrated by | Jira status |
|---|---|---|---|
| SPP-26 | T1 Jira and repository setup | Dakshitha | Done |
| SPP-8 | O1 Environment | Dakshitha; independently checked by Chaitanya (approved PR #3) | Done |
| SPP-9 | D1 Dataset | Divya | Done |
| SPP-10 | D2 Validation | Divya | Done |
| SPP-11 | D3 Partitions | Divya | Done |
| SPP-12 | M1 Baselines | Prabhanjan | Done |
| SPP-13 | M2 Experiments | Prabhanjan | Done |
| SPP-14 | M3 Model package | Prabhanjan | Done |
| SPP-15 | A1 API | Chaitanya | Done |
| SPP-27 | T2 Sprint 1 evidence | Dakshitha | In Review (see below) |

Each story was merged through a pull request that passed CI. Reviewers are recorded on the pull requests in GitHub.

## Outcome
- Planned: 26 story points.
- Completed points: taken from Jira's Sprint report once the sprint is closed, and added in a follow-up commit. They are not estimated here.
- Acceptance criteria not met: none known for the nine Done items. T2 is open only until this document is approved and merged.
- Carried-over issues: none expected. T2 will be Done before the sprint is closed; if it is not, it is recorded here as carried over.

## Why the sprint ran 7 Oct to 8 Oct
The original plan was a longer window (6 Oct to 19 Oct). The team compressed the schedule and carried out Sprint 1 on 7 and 8 October, so the sprint dates were set to the days the work really happened.

## Status of T2 (this task)
T2 is In Review. Pull request #6 was opened for review and Divya's corrections have been applied. T2 moves to Done only when the pull request is approved and merged. This document will not say otherwise.

## Retrospective
What went well:
- CI caught a Windows-only package and it was fixed before merging.
- Every story went through a pull request; the O1 setup was checked independently by Chaitanya on a second machine.
- CI was shown to fail on a deliberately broken test and pass after the fix (E10b, E10c).

What could improve:
- The first Jira project was created as Kanban, so sprints were missing and the project was recreated as Scrum.
- Two people wrote overlapping data modules, which caused merge conflicts.
- Some teammates received Jira and GitHub access late.

Actions for Sprint 2:
1. Agree who owns each module before anyone starts coding. Owner: Dakshitha. Deadline: 9 Oct 2026, at Sprint 2 planning.
2. Each member updates their own Jira card when they start and finish work. Owner: each team member. Deadline: from 9 Oct 2026, checked at every stand-up.

## Evidence
- E06: evidence/E06_2026-10-08_sprint1-board.png (Sprint 1 board)
- E07: Sprint 1 burndown and velocity report. To be added after the sprint is closed.