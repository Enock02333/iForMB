
# Runtime and resource expectations

The following are planning ranges. Actual time depends on CPU, storage, and GPU. The paper runs were completed on a Windows workstation with an NVIDIA GTX 1650 for the matched external-baseline stage.

| Workflow | Environment | Planning range |
|---|---|---:|
| Unit and integration tests | CPU | under 1 minute |
| Synthetic development pipeline | CPU | minutes to tens of minutes, depending on record count |
| HDFS conventional, five seeds | CPU | tens of minutes to several hours |
| HDFS unseen-type, 45 fits | CPU | several hours |
| BGL chronological, five seeds | CPU | approximately 1–4 hours |
| DeepLog and LogRobust matched stage | GPU recommended | approximately 2–8 hours |

Record measured runtimes in provenance files for every new run; do not treat these ranges as benchmark measurements.
