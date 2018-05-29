# Click prediction on Criteo 1Tb dataset

##### By: Traci Lim

---

This repository holds a report in the form of a jupyter notebook, which focused on the prediction of the click through rate of display ads, while managing the following objectives:

- Utilize Microsoft Azure HDInsight clusters for large-scale machine learning.
- Briefly summarize the key concepts behind algorithms used.
- Address the problem of class imbalance by resampling methods.
- Give a concise experimental analysis with variations in data and  number of worker nodes/partitions, for understanding the scalability and  distributed performance of algorithms in MLlib.
- Evaluate results and conclude findings.



---

### Experimental Setup

To configure the Spark application to use up as much of the cluster setup as possible, we ran a long series of tests on different configurations, and finalized the scores below. 

Cluster specifications table:

|         Cluster type         | Cores per Head Node | Cores per Worker Node | Memory per Node (Head, Worker) | Total Cores |    No. of Nodes    |
| :--------------------------: | :-----------------: | :-------------------: | :----------------------------: | :---------: | :----------------: |
| Spark 2.1 on Linux (HDI 3.6) |          4          |           8           |         (28GiB, 56GiB)         |     40      | 2x Head, 4x Worker |

Spark configuration options table:


| Master                                 | Yarn Cilent |
| -------------------------------------- | ----------- |
| `yarn.nodemanager.resource.memory-mb`  | 56320mb     |
| `yarn.nodemanager.resource.cpu-vcores` | 7           |
| `num-executors`                        | 6           |
| `executor-cores`                       | 4           |
| `executor-memory`                      | 11060mb     |
| `spark.dynamicAllocation.enabled`      | true        |
| `spark.shuffle.service.enabled`        | true        |
| `spark.driver.cores`                   | 4           |
| `spark.driver.memory`                  | 11060mb     |

---

